from dataclasses import dataclass, field
from typing import Dict, Tuple, Sequence, Optional, List
import numpy as np

StateKey = Tuple[int, ...]


def _copy_q(Q: Dict[StateKey, np.ndarray]) -> Dict[StateKey, np.ndarray]:
    return {s: v.copy() for s, v in Q.items()}


@dataclass
class RLMetricsTrackerTabular:
    num_actions: int

    q_prev: Optional[Dict[StateKey, np.ndarray]] = None
    policy_prev: Optional[Dict[StateKey, int]] = None

    q_deltas: List[float] = field(default_factory=list)
    policy_deltas: List[float] = field(default_factory=list)
    action_histories: List[np.ndarray] = field(default_factory=list)

    def _q_convergence(
        self,
        Q_prev: Dict[StateKey, np.ndarray],
        Q: Dict[StateKey, np.ndarray],
    ) -> float:
        all_states = set(Q_prev.keys()) | set(Q.keys())
        delta = 0.0
        zero = np.zeros(self.num_actions, dtype=float)

        for s in all_states:
            q_old = Q_prev.get(s, zero)
            q_new = Q.get(s, zero)

            if q_old.shape[0] != self.num_actions:
                q_old = np.resize(q_old, self.num_actions)
            if q_new.shape[0] != self.num_actions:
                q_new = np.resize(q_new, self.num_actions)

            delta_state = float(np.max(np.abs(q_new - q_old)))
            if delta_state > delta:
                delta = delta_state

        return delta

    def _greedy_policy(
        self,
        Q: Dict[StateKey, np.ndarray],
    ) -> Dict[StateKey, int]:
        policy: Dict[StateKey, int] = {}
        for s, q_values in Q.items():
            if q_values.shape[0] != self.num_actions:
                q_values = np.resize(q_values, self.num_actions)
            best_action = int(np.argmax(q_values))
            policy[s] = best_action
        return policy

    def _policy_convergence(
        self,
        policy_prev: Dict[StateKey, int],
        policy: Dict[StateKey, int],
    ) -> Optional[float]:
        common_states = set(policy_prev.keys()) & set(policy.keys())
        if not common_states:
            return None

        changes = sum(policy_prev[s] != policy[s] for s in common_states)
        return float(changes) / float(len(common_states))

    def _action_distribution(
        self,
        actions: Sequence[int],
    ) -> np.ndarray:
        actions_arr = np.asarray(actions, dtype=int)
        counts = np.bincount(actions_arr, minlength=self.num_actions)
        return counts.astype(float)

    def update(
        self,
        Q: Dict[StateKey, np.ndarray],
        episode_actions: Sequence[int],
    ) -> tuple[Optional[float], Optional[float], np.ndarray]:
        
        # delta Q
        delta_q: Optional[float] = None
        if self.q_prev is not None:
            delta_q = self._q_convergence(self.q_prev, Q)
            self.q_deltas.append(delta_q)
        self.q_prev = _copy_q(Q)

        # delta policy
        delta_policy: Optional[float] = None
        policy = self._greedy_policy(Q)
        if self.policy_prev is not None:
            delta_policy = self._policy_convergence(self.policy_prev, policy)
            if delta_policy is not None:
                self.policy_deltas.append(delta_policy)
        self.policy_prev = policy

        # action histogram
        action_counts = self._action_distribution(episode_actions)
        self.action_histories.append(action_counts)

        return delta_q, delta_policy, action_counts
