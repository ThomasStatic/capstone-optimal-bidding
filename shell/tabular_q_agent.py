
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Sequence, Union
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# Our State object currently produces Pandas row of discrete values, so needs to be converted to a hashable type
StateKey = Tuple[int, ...]
StateVec = ArrayLike  # e.g., np.ndarray or pd.Series

@dataclass
class TabularQLearningAgent:
    num_actions: int
    alpha: float = 0.1  # Learning rate
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995  # Decay rate for exploration

    Q: Dict[StateKey, np.ndarray] = field(default_factory=dict, init=False)
    _rng: np.random.Generator = field(init=False) # used for random exploration

    def __post_init__(self):
        self._rng = np.random.default_rng()

    def state_to_key(self, state_vec: StateVec) -> StateKey:
        """Convert a state representation to a hashable StateKey."""
        if isinstance(state_vec, pd.Series):
            vals = state_vec.to_numpy()
        else:
            vals = np.asarray(state_vec)
        
        return tuple(int(v) for v in vals)
    
    def _ensure_state(self, key: StateKey) -> None:
        if key not in self.Q:
            self.Q[key] = np.zeros(self.num_actions, dtype=float)
    
    # epsilon-greedy action selection
    def select_action(self, state_key: StateKey) -> int:
        self._ensure_state(state_key)

        # Exploration
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.num_actions))
        
        # Exploitation
        else:
            q_values = self.Q[state_key]

            # tie breaking
            max_q = np.max(q_values)
            best_actions = np.flatnonzero(q_values == max_q)
            return int(self._rng.choice(best_actions))
        
    def get_softmax_action_probs(self, state_key: StateKey, temperature: float | None = None) -> np.ndarray:
        self._ensure_state(state_key)
        q_values = self.Q[state_key]
        
        if temperature is None:
            temperature = self.temperature_from_qgap(q_values, p_star=0.8)

        # Temperature must be positive for softmax, fallback to epsilon greedy
        if temperature <= 0:
            probs = np.zeros_like(q_values, dtype=float)
            probs[np.argmax(q_values)] = 1.0
            return probs
        
        scaled = q_values / temperature
        scaled -= np.max(scaled)
        exp_q = np.exp(scaled)
        return exp_q / np.sum(exp_q)
    
    def select_softmax_action(self, state_key: StateKey, temperature: float | None = None) -> int:
        probs = self.get_softmax_action_probs(state_key, temperature)
        return int(self._rng.choice(self.num_actions, p=probs))

        
    def update_q_table(self, state_key: StateKey, action: int, reward: float, next_state_key: StateKey, done: bool) -> None:
        self._ensure_state(state_key)
        quality_current = self.Q[state_key][action]

        if done or next_state_key is None:
            target = reward
        else:
            self._ensure_state(next_state_key)
            target = reward + self.gamma * np.max(self.Q[next_state_key])

        self.Q[state_key][action] = quality_current + self.alpha * (target - quality_current)

    def extract_softmax_policy(self, temperature: float = 1.0) -> Dict[StateKey, np.ndarray | int]:
        policy: Dict[StateKey, np.ndarray | int] = {}
        for state_key in self.Q.keys():
            probs = self.get_softmax_action_probs(state_key, temperature)
            policy[state_key] = int(np.argmax(probs)) # only do this for deterministic policy
        return policy

    def decay_epsilon(self) -> None:
        self.epsilon *= self.epsilon_decay

    def temperature_from_qgap(
        self,
        q_values: np.ndarray, 
        p_star: float = 0.8, 
        t_min: float = 0.05, 
        t_max:float = 5.0
    ) -> float:
        q = np.asarray(q_values, dtype=float)
        if q.size < 2:
            return 1.0  # No gap if only one action
        
        top2 = np.partition(q, -2)[-2:]
        q2, q1 = np.sort(top2)  # q1 >= q2
        gap = float(q1 - q2)

        if gap <= 1e-12:
            return t_max
        
        denom = np.log(p_star / (1.0 - p_star))
        T = gap / denom
        return float(np.clip(T, t_min, t_max))