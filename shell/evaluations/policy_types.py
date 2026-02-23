from typing import Protocol

from shell.agent_interface import BiddingAgent, Observation

# Re-export for consumers that expect Policy types
PolicyObs = Observation  # Observation extends PolicyObs (timestamp required)
Policy = BiddingAgent  # Standardized interface: act(obs) -> int
