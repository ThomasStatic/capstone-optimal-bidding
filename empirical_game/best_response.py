from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from empirical_game.profile_space import JointProfile, ProfileSpace, with_unilateral_deviation


DecisionRule = Literal["plain", "conservative"]


@dataclass(frozen=True)
class BestResponseConfig:
    epsilon: float = 0.0
    decision_rule: DecisionRule = "plain"
    confidence: float = 0.95


def _z_value(confidence: float) -> float:
    return 1.96 if confidence >= 0.95 else 1.645


def deviation_table(
    *,
    profile_space: ProfileSpace,
    rollout_by_profile: dict[str, np.ndarray],
    br_config: BestResponseConfig,
) -> pd.DataFrame:
    """Enumerate unilateral deviations and estimate gain statistics."""
    z = _z_value(br_config.confidence)
    rows: list[dict] = []

    for profile in profile_space.profiles:
        base_key = profile.profile_id
        base_rollouts = rollout_by_profile[base_key]

        for idx, agent_id in enumerate(profile.agent_ids):
            from_policy = profile.policy_ids[idx]
            candidate_ids = list(profile_space.policy_lookup[agent_id].keys())
            for to_policy in candidate_ids:
                if to_policy == from_policy:
                    continue

                dev_profile = with_unilateral_deviation(
                    profile,
                    agent_id=agent_id,
                    to_policy_id=to_policy,
                )
                dev_key = dev_profile.profile_id
                dev_rollouts = rollout_by_profile[dev_key]

                gain_samples = dev_rollouts[:, idx] - base_rollouts[:, idx]
                gain_mean = float(gain_samples.mean())
                gain_se = float(gain_samples.std(ddof=1) / np.sqrt(len(gain_samples))) if len(gain_samples) > 1 else 0.0
                gain_lcb = float(gain_mean - z * gain_se)

                if br_config.decision_rule == "plain":
                    is_profitable = gain_mean > br_config.epsilon
                elif br_config.decision_rule == "conservative":
                    is_profitable = gain_lcb > br_config.epsilon
                else:
                    raise ValueError(f"Unknown decision rule: {br_config.decision_rule}")

                from_meta = profile_space.policy_lookup[agent_id][from_policy].metadata
                to_meta = profile_space.policy_lookup[agent_id][to_policy].metadata

                rows.append(
                    {
                        "base_profile_id": base_key,
                        "deviation_profile_id": dev_key,
                        "agent_id": int(agent_id),
                        "from_policy_id": from_policy,
                        "to_policy_id": to_policy,
                        "from_policy_kind": from_meta.get("policy_kind", "unknown"),
                        "to_policy_kind": to_meta.get("policy_kind", "unknown"),
                        "gain_mean": gain_mean,
                        "gain_se": gain_se,
                        "gain_lcb": gain_lcb,
                        "epsilon": float(br_config.epsilon),
                        "decision_rule": br_config.decision_rule,
                        "is_profitable": bool(is_profitable),
                    }
                )

    return pd.DataFrame(rows)


def best_reply_edges(deviation_df: pd.DataFrame) -> pd.DataFrame:
    """Directed edges induced by unilateral approximate strict best replies."""
    edges = deviation_df.loc[deviation_df["is_profitable"]].copy()
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "agent_id",
                "from_policy_id",
                "to_policy_id",
                "gain_mean",
                "gain_lcb",
                "decision_rule",
                "epsilon",
            ]
        )

    edges = edges.rename(
        columns={
            "base_profile_id": "source",
            "deviation_profile_id": "target",
        }
    )
    return edges[
        [
            "source",
            "target",
            "agent_id",
            "from_policy_id",
            "to_policy_id",
            "gain_mean",
            "gain_lcb",
            "decision_rule",
            "epsilon",
        ]
    ].reset_index(drop=True)


def best_response_gaps(deviation_df: pd.DataFrame) -> pd.DataFrame:
    """Best-response gap at each profile and agent: max estimated unilateral gain."""
    if deviation_df.empty:
        return pd.DataFrame(columns=["profile_id", "agent_id", "best_response_gap"])

    grouped = deviation_df.groupby(["base_profile_id", "agent_id"], as_index=False).agg(
        best_response_gap=("gain_mean", "max")
    )
    grouped["profile_id"] = grouped["base_profile_id"]
    grouped = grouped.drop(columns=["base_profile_id"])
    return grouped[["profile_id", "agent_id", "best_response_gap"]]


def sink_profiles(
    *,
    all_profile_ids: list[str],
    edge_df: pd.DataFrame,
) -> list[str]:
    """Approximate equilibrium sinks = nodes with no outgoing profitable deviations."""
    if edge_df.empty:
        return sorted(all_profile_ids)
    outgoing = set(edge_df["source"].tolist())
    return sorted([pid for pid in all_profile_ids if pid not in outgoing])


def profitable_deviations_from_profile(
    *,
    profile_id: str,
    deviation_df: pd.DataFrame,
) -> pd.DataFrame:
    return deviation_df[(deviation_df["base_profile_id"] == profile_id) & (deviation_df["is_profitable"])].copy()
