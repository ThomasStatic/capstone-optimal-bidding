
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Sequence, Union
import numpy as np
import pandas as pd

# Our State object currently produces Pandas row of discrete values, so needs to be converted to a hashable type
StateKey = Tuple[int, ...]
StateVec = Union[np.ndarray, pd.Series]

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
            vals = state_vec
        
        return tuple(int(v) for v in vals)
    
    def _ensure_state(self, key: StateKey) -> None:
        if key not in self.Q:
            self.Q[key] = np.zeros(self.num_actions, dtype=float)
    

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
        
    def update_q_table(self, state_key: StateKey, action: int, reward: float, next_state_key: StateKey, done: bool) -> None:
        self._ensure_state(state_key)
        
        quality_current = self.Q[state_key][action]

        if done or next_state_key is None:
            target = reward
        else:
            self._ensure_state(next_state_key)
            target = reward + self.gamma * np.max(self.Q[next_state_key])

        self.Q[state_key][action] += quality_current + self.alpha * (target - quality_current)

    def decay_epsilon(self) -> None:
        self.epsilon *= self.epsilon_decay