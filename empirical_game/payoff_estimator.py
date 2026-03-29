from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd

from empirical_game.policy_wrappers import PolicyWrapper
from empirical_game.profile_space import JointProfile, ProfileSpace


class JointProfileSimulator(Protocol):
    """Adapter interface to evaluate a joint policy profile in the project simulator."""

    n_agents: int

    def evaluate_profile_once(self, joint_policies: Sequence[PolicyWrapper], seed: int) -> np.ndarray:
        """Return one Monte Carlo sample payoff vector (shape: [n_agents])."""
        ...


@dataclass(frozen=True)
class MonteCarloConfig:
    n_rollouts: int = 16
    base_seed: int = 123
    confidence: float = 0.95
    use_common_random_numbers: bool = True
    show_progress: bool = True
    profile_progress_every: int = 1
    rollout_progress_every: int = 0


@dataclass
class MonteCarloPayoffEstimator:
    simulator: JointProfileSimulator
    config: MonteCarloConfig

    def rollout_seeds(self) -> list[int]:
        if self.config.n_rollouts <= 0:
            raise ValueError("n_rollouts must be > 0")

        if self.config.use_common_random_numbers:
            return [int(self.config.base_seed + k) for k in range(self.config.n_rollouts)]

        rng = np.random.default_rng(self.config.base_seed)
        return [int(x) for x in rng.integers(0, 2**31 - 1, size=self.config.n_rollouts)]

    def estimate_profile_rollouts(
        self,
        *,
        profile: JointProfile,
        profile_space: ProfileSpace,
        seeds: Sequence[int],
        rollout_progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        policies = profile_space.policies_for_profile(profile)
        rows: list[np.ndarray] = []
        total_rollouts = len(seeds)
        for rollout_idx, s in enumerate(seeds, start=1):
            payoff = self.simulator.evaluate_profile_once(policies, int(s))
            if payoff.shape != (self.simulator.n_agents,):
                raise ValueError(
                    f"Simulator payoff shape mismatch: expected {(self.simulator.n_agents,)}, got {payoff.shape}"
                )
            rows.append(payoff.astype(float))
            if rollout_progress_callback is not None:
                rollout_progress_callback(rollout_idx, total_rollouts)
        return np.vstack(rows)

    def estimate_profile_table(
        self,
        *,
        profile: JointProfile,
        rollouts: np.ndarray,
    ) -> pd.DataFrame:
        n = rollouts.shape[0]
        means = rollouts.mean(axis=0)
        std_err = rollouts.std(axis=0, ddof=1) / np.sqrt(max(n, 1)) if n > 1 else np.zeros_like(means)

        z = 1.96 if self.config.confidence >= 0.95 else 1.645
        ci_low = means - z * std_err
        ci_high = means + z * std_err

        rows = []
        for idx, agent_id in enumerate(profile.agent_ids):
            rows.append(
                {
                    "profile_id": profile.profile_id,
                    "agent_id": int(agent_id),
                    "mean_payoff": float(means[idx]),
                    "std_err": float(std_err[idx]),
                    "ci_low": float(ci_low[idx]),
                    "ci_high": float(ci_high[idx]),
                    "n_rollouts": int(n),
                }
            )
        return pd.DataFrame(rows)

    def estimate_all_profiles(
        self,
        profile_space: ProfileSpace,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[int]]:
        seeds = self.rollout_seeds()

        payoff_tables: list[pd.DataFrame] = []
        rollout_by_profile: dict[str, np.ndarray] = {}

        total_profiles = len(profile_space.profiles)
        total_rollouts = total_profiles * len(seeds)
        completed_rollouts = 0
        t0 = time()

        if self.config.show_progress:
            print(
                f"[EmpiricalGame] Starting payoff estimation: "
                f"profiles={total_profiles}, rollouts/profile={len(seeds)}, total_rollouts={total_rollouts}",
                flush=True,
            )

        for profile_idx, profile in enumerate(profile_space.profiles, start=1):
            if self.config.show_progress:
                print(
                    f"[EmpiricalGame] Profile {profile_idx}/{total_profiles} -> {profile.profile_id}",
                    flush=True,
                )

            def _on_rollout(rollout_idx: int, n_rollouts: int) -> None:
                nonlocal completed_rollouts
                completed_rollouts += 1

                step = int(self.config.rollout_progress_every)
                if step <= 0:
                    return
                if (rollout_idx % step) != 0 and rollout_idx != n_rollouts:
                    return

                elapsed = max(time() - t0, 1e-9)
                rate = completed_rollouts / elapsed
                remaining_rollouts = max(total_rollouts - completed_rollouts, 0)
                eta_sec = remaining_rollouts / rate if rate > 0 else float("inf")
                print(
                    f"[EmpiricalGame]   rollout {rollout_idx}/{n_rollouts} | "
                    f"global={completed_rollouts}/{total_rollouts} | "
                    f"elapsed={elapsed:.1f}s | eta={eta_sec:.1f}s",
                    flush=True,
                )

            rollout_matrix = self.estimate_profile_rollouts(
                profile=profile,
                profile_space=profile_space,
                seeds=seeds,
                rollout_progress_callback=_on_rollout,
            )
            rollout_by_profile[profile.profile_id] = rollout_matrix
            payoff_tables.append(self.estimate_profile_table(profile=profile, rollouts=rollout_matrix))

            if self.config.show_progress:
                every = max(1, int(self.config.profile_progress_every))
                if (profile_idx % every) == 0 or profile_idx == total_profiles:
                    elapsed = max(time() - t0, 1e-9)
                    rate = profile_idx / elapsed
                    remaining_profiles = max(total_profiles - profile_idx, 0)
                    eta_sec = remaining_profiles / rate if rate > 0 else float("inf")
                    print(
                        f"[EmpiricalGame] Completed profile {profile_idx}/{total_profiles} | "
                        f"elapsed={elapsed:.1f}s | eta={eta_sec:.1f}s",
                        flush=True,
                    )

        if self.config.show_progress:
            elapsed = time() - t0
            print(f"[EmpiricalGame] Payoff estimation complete in {elapsed:.1f}s", flush=True)

        return pd.concat(payoff_tables, ignore_index=True), rollout_by_profile, seeds
