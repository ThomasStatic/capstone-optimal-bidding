"""
Standardized agent interface for single- and multi-agent bidding.

All agents (RL, baselines, saved policies) implement BiddingAgent so that
training loops, evaluation, and demand perturbation can use the same call pattern:
  action_idx = agent.act(obs)
"""
from __future__ import annotations

from typing import Any, Protocol, TypedDict, TypeAlias

import numpy as np
import pandas as pd

# Hashable state key used by tabular Q-learning and saved policies
StateKey: TypeAlias = tuple[int, ...]


class Observation(TypedDict, total=False):
    """
    Unified observation passed to agent.act(obs).

    Required for all agents:
      timestamp: current step time (UTC)

    Optional (used by RL and saved-policy agents):
      state_vec: discretized state row (array or Series)
      state_key: hashable key from state_vec (avoids repeated conversion)

    Optional (used during training):
      temperature: for softmax action selection
    """
    timestamp: pd.Timestamp
    state_vec: np.ndarray | pd.Series
    state_key: StateKey
    temperature: float | None


def state_vec_to_key(state_vec: np.ndarray | pd.Series) -> StateKey:
    """Convert state row to hashable key. Shared by agents and evaluation."""
    if hasattr(state_vec, "to_numpy"):
        vals = state_vec.to_numpy()
    else:
        vals = np.asarray(state_vec)
    return tuple(int(v) for v in vals)


class BiddingAgent(Protocol):
    """
    Standard interface for any bidding agent (RL, baseline, or saved policy).

    Implementations must provide act(obs). Learning agents may also implement
    update() and extract_policy() for training and checkpointing.
    """

    def act(self, obs: Observation) -> int:
        """
        Return a single discrete action index for the given observation.

        obs must contain at least "timestamp". RL/saved-policy agents use
        "state_key" or "state_vec" to look up the action.
        """
        ...


class LearningAgent(BiddingAgent, Protocol):
    """Optional interface for agents that learn (e.g. tabular Q-learning)."""

    def state_to_key(self, state_vec: Any) -> StateKey:
        """Convert state vector to hashable key."""
        ...

    def update(
        self,
        obs: Observation,
        action: int,
        reward: float,
        next_obs: Observation | None,
        done: bool,
    ) -> None:
        """Update internal model (e.g. Q-table). No-op for non-learning agents."""
        ...

    def extract_policy(self, **kwargs: Any) -> dict[StateKey, int] | list[dict[StateKey, int]]:
        """
        Return a deterministic policy for evaluation/saving.

        For single-agent: dict state_key -> action_idx.
        For multi-agent: list of such dicts (one per agent).
        """
        ...


class SavedPolicyAgent:
    """
    Wrapper that implements BiddingAgent using a saved policy dict (and optional Q-table fallback).
    Used for demand perturbation and any evaluation of a checkpoint without loading the full RL agent.
    """

    def __init__(
        self,
        policy_map: dict[StateKey, int],
        *,
        q_table: dict[StateKey, np.ndarray] | None = None,
        fallback_action: int = 0,
    ) -> None:
        self.policy_map = policy_map
        self.q_table = q_table
        self.fallback_action = fallback_action

    def act(self, obs: Observation) -> int:
        state_key = obs.get("state_key")
        if state_key is None and "state_vec" in obs:
            state_key = state_vec_to_key(obs["state_vec"])
        if state_key is None:
            return self.fallback_action
        if state_key in self.policy_map:
            return int(self.policy_map[state_key])
        if self.q_table is not None and state_key in self.q_table:
            return int(np.argmax(self.q_table[state_key]))
        return self.fallback_action


def load_saved_agents(
    policy_path: str,
    q_table_path: str | None = None,
    *,
    fallback_action: int = 0,
) -> list[SavedPolicyAgent]:
    """
    Load saved policy (and optional Q-table) from disk.
    policy_path may contain a single dict (state_key -> action) or a list of dicts (multi-agent).
    Returns a list of SavedPolicyAgent (one per agent).
    """
    import pickle

    with open(policy_path, "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        policy_maps = [raw]
    elif isinstance(raw, list) and all(isinstance(x, dict) for x in raw):
        policy_maps = raw
    else:
        raise ValueError(f"Expected policy file to contain dict or list of dicts, got {type(raw)}")

    q_tables: list[dict[StateKey, np.ndarray] | None] = [None] * len(policy_maps)
    if q_table_path:
        try:
            with open(q_table_path, "rb") as f:
                qt_raw = pickle.load(f)
            if isinstance(qt_raw, list) and len(qt_raw) == len(policy_maps):
                q_tables = list(qt_raw)
            elif isinstance(qt_raw, dict):
                q_tables = [qt_raw] * len(policy_maps)
        except FileNotFoundError:
            pass

    return [
        SavedPolicyAgent(
            policy_map=pm,
            q_table=qt,
            fallback_action=fallback_action,
        )
        for pm, qt in zip(policy_maps, q_tables)
    ]
