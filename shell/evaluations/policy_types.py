from typing import Protocol, TypedDict
import pandas as pd


class PolicyObs(TypedDict):
    timestamp: pd.Timestamp
    
class Policy(Protocol):
    def act(self, obs: PolicyObs) -> int:
        ...
