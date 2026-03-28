from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from shell.action_space import ActionSpace
from shell.agent_interface import Observation, StateKey


@dataclass(frozen=True)
class PolicyWrapper:
    """Uniform deterministic policy interface for reduced empirical games."""

    policy_id: str
    agent_id: int
    act_fn: Callable[[Observation], int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def act(self, obs: Observation) -> int:
        return int(self.act_fn(obs))


@dataclass(frozen=True)
class PerturbationSpec:
    """Deterministic transform on discrete action bins."""

    name: str
    price_shift_bins: int = 0
    quantity_shift_bins: int = 0


def action_index_with_bin_shift(
    action_index: int,
    action_space: ActionSpace,
    *,
    price_shift_bins: int = 0,
    quantity_shift_bins: int = 0,
) -> int:
    """Apply deterministic bin shifts with clipping to valid action-bin bounds."""
    price_bin, qty_bin = action_space.index_to_pair(int(action_index))

    shifted_price_bin = max(0, min(action_space.n_price_bins - 1, price_bin + int(price_shift_bins)))
    shifted_qty_bin = max(0, min(action_space.n_quantity_bins - 1, qty_bin + int(quantity_shift_bins)))

    return int(action_space.pair_to_index(shifted_price_bin, shifted_qty_bin))


def make_policy_wrapper_from_callable(
    *,
    policy_id: str,
    agent_id: int,
    act_fn: Callable[[Observation], int],
    metadata: Mapping[str, Any] | None = None,
) -> PolicyWrapper:
    return PolicyWrapper(
        policy_id=policy_id,
        agent_id=agent_id,
        act_fn=act_fn,
        metadata=metadata or {},
    )


def make_policy_wrapper_from_map(
    *,
    policy_id: str,
    agent_id: int,
    policy_map: Mapping[StateKey, int],
    fallback_action: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> PolicyWrapper:
    """Create a deterministic wrapper from a saved state->action map."""

    def _act(obs: Observation) -> int:
        key = obs.get("state_key")
        if key is not None and key in policy_map:
            return int(policy_map[key])
        return int(fallback_action)

    return make_policy_wrapper_from_callable(
        policy_id=policy_id,
        agent_id=agent_id,
        act_fn=_act,
        metadata=metadata,
    )


def make_perturbed_policy_wrapper(
    *,
    base_policy: PolicyWrapper,
    action_space: ActionSpace,
    perturbation: PerturbationSpec,
    policy_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PolicyWrapper:
    """Return a deterministic wrapper that applies a local action-bin perturbation."""

    new_policy_id = policy_id or f"{base_policy.policy_id}__{perturbation.name}"
    merged_meta = dict(base_policy.metadata)
    merged_meta.update(metadata or {})
    merged_meta.update(
        {
            "source_policy_id": base_policy.policy_id,
            "policy_kind": "perturbation",
            "perturbation_name": perturbation.name,
            "price_shift_bins": perturbation.price_shift_bins,
            "quantity_shift_bins": perturbation.quantity_shift_bins,
        }
    )

    def _act(obs: Observation) -> int:
        base_action = int(base_policy.act(obs))
        return action_index_with_bin_shift(
            base_action,
            action_space,
            price_shift_bins=perturbation.price_shift_bins,
            quantity_shift_bins=perturbation.quantity_shift_bins,
        )

    return make_policy_wrapper_from_callable(
        policy_id=new_policy_id,
        agent_id=base_policy.agent_id,
        act_fn=_act,
        metadata=merged_meta,
    )


def default_local_perturbations() -> list[PerturbationSpec]:
    return [
        PerturbationSpec(name="price_up_1", price_shift_bins=1, quantity_shift_bins=0),
        PerturbationSpec(name="price_down_1", price_shift_bins=-1, quantity_shift_bins=0),
        PerturbationSpec(name="quantity_up_1", price_shift_bins=0, quantity_shift_bins=1),
        PerturbationSpec(name="quantity_down_1", price_shift_bins=0, quantity_shift_bins=-1),
    ]
