
from dataclasses import dataclass
from typing import Dict


@dataclass
class TabularQLearningAgent:
    alpha: float = 0.1  # Learning rate
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995  # Decay rate for exploration

    