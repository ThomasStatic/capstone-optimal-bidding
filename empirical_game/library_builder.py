from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from shell.action_space import ActionSpace

from empirical_game.policy_wrappers import (
    PolicyWrapper,
    default_local_perturbations,
    make_perturbed_policy_wrapper,
)


@dataclass(frozen=True)
class LibraryBuildOptions:
    include_final_policy: bool = True
    include_baselines: bool = True
    include_local_perturbations: bool = True


def deduplicate_policy_library(policies: Iterable[PolicyWrapper]) -> list[PolicyWrapper]:
    """Deduplicate by policy_id while preserving first occurrence order."""
    seen: set[str] = set()
    out: list[PolicyWrapper] = []
    for p in policies:
        if p.policy_id in seen:
            continue
        seen.add(p.policy_id)
        out.append(p)
    return out


def build_policy_library_for_agent(
    *,
    agent_id: int,
    final_policy: PolicyWrapper,
    action_space: ActionSpace,
    baseline_policies: Iterable[PolicyWrapper] | None = None,
    include_local_perturbations: bool = True,
) -> list[PolicyWrapper]:
    """Build a finite policy library for one agent."""
    policies: list[PolicyWrapper] = [final_policy]

    if baseline_policies:
        policies.extend(list(baseline_policies))

    if include_local_perturbations:
        for spec in default_local_perturbations():
            policies.append(
                make_perturbed_policy_wrapper(
                    base_policy=final_policy,
                    action_space=action_space,
                    perturbation=spec,
                    metadata={"agent_id": agent_id},
                )
            )

    return deduplicate_policy_library(policies)


def build_policy_libraries(
    *,
    final_policies: Mapping[int, PolicyWrapper],
    action_space: ActionSpace,
    baselines_by_agent: Mapping[int, list[PolicyWrapper]] | None = None,
    options: LibraryBuildOptions | None = None,
) -> dict[int, list[PolicyWrapper]]:
    """Build finite reduced-game policy libraries for all agents."""
    cfg = options or LibraryBuildOptions()

    libraries: dict[int, list[PolicyWrapper]] = {}
    for agent_id, final_policy in final_policies.items():
        baseline_list = []
        if cfg.include_baselines and baselines_by_agent is not None:
            baseline_list = baselines_by_agent.get(agent_id, [])

        if not cfg.include_final_policy:
            raise ValueError("include_final_policy=False is not supported for reduced-game construction.")

        libraries[agent_id] = build_policy_library_for_agent(
            agent_id=agent_id,
            final_policy=final_policy,
            action_space=action_space,
            baseline_policies=baseline_list,
            include_local_perturbations=cfg.include_local_perturbations,
        )

    return libraries
