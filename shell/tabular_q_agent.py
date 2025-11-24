
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Sequence
import numpy as np
import pandas as pd

# Our State object currently produces Pandas row of discrete values, so needs to be converted to a hashable type
StateKey = Tuple[int, ...]

@dataclass
class TabularQLearningAgent:
    alpha: float = 0.1  # Learning rate
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995  # Decay rate for exploration

    Q: Dict[StateKey, np.ndarray] = field(default_factory=dict, init=False)
    _rng: np.random.Generator = field(init=False) # used for random exploration

    def __post_init__(self):
        self._rng = np.random.default_rng()

    def state_to_key(self, state_row: pd.Series, columns: Optional[Sequence[str]] = None) -> StateKey:
        """Convert a state representation to a hashable StateKey."""
        if columns is None:
            vals = state_row.to_numpy()
        else:
            vals = state_row.loc[list(columns)].to_numpy()
        
        return tuple(int(v) for v in vals)
    
    
    

        