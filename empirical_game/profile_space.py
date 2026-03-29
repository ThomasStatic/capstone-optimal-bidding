from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from empirical_game.policy_wrappers import PolicyWrapper


@dataclass(frozen=True)
class JointProfile:
    """Reduced-game joint profile represented by policy IDs in agent order."""

    agent_ids: tuple[int, ...]
    policy_ids: tuple[str, ...]

    @property
    def profile_id(self) -> str:
        return "|".join(f"a{aid}:{pid}" for aid, pid in zip(self.agent_ids, self.policy_ids))


@dataclass(frozen=True)
class ProfileSpace:
    """Container for all reduced-game joint profiles and policy lookup tables."""

    profiles: list[JointProfile]
    policy_lookup: dict[int, dict[str, PolicyWrapper]]

    def policies_for_profile(self, profile: JointProfile) -> tuple[PolicyWrapper, ...]:
        return tuple(self.policy_lookup[aid][pid] for aid, pid in zip(profile.agent_ids, profile.policy_ids))

    def profile_by_id(self, profile_id: str) -> JointProfile:
        for p in self.profiles:
            if p.profile_id == profile_id:
                return p
        raise KeyError(f"Unknown profile_id: {profile_id}")


def build_profile_space(libraries: Mapping[int, Sequence[PolicyWrapper]]) -> ProfileSpace:
    """Enumerate all joint profiles from per-agent finite policy libraries."""
    agent_ids = tuple(sorted(libraries.keys()))

    policy_lookup: dict[int, dict[str, PolicyWrapper]] = {}
    policy_id_lists: list[list[str]] = []

    for agent_id in agent_ids:
        per_agent = list(libraries[agent_id])
        lookup = {p.policy_id: p for p in per_agent}
        if len(lookup) != len(per_agent):
            raise ValueError(f"Agent {agent_id} has duplicate policy IDs in library.")
        policy_lookup[agent_id] = lookup
        policy_id_lists.append(list(lookup.keys()))

    profiles = [
        JointProfile(agent_ids=agent_ids, policy_ids=tuple(policy_ids))
        for policy_ids in product(*policy_id_lists)
    ]

    return ProfileSpace(profiles=profiles, policy_lookup=policy_lookup)


def with_unilateral_deviation(
    profile: JointProfile,
    *,
    agent_id: int,
    to_policy_id: str,
) -> JointProfile:
    """Return profile after replacing one agent's policy."""
    if agent_id not in profile.agent_ids:
        raise KeyError(f"Agent {agent_id} not in profile")

    idx = profile.agent_ids.index(agent_id)
    new_ids = list(profile.policy_ids)
    new_ids[idx] = to_policy_id
    return JointProfile(agent_ids=profile.agent_ids, policy_ids=tuple(new_ids))
