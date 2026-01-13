from dataclasses import dataclass
from typing import Optional
import pandas as pd
from shell.action_space import ActionSpace
from shell.evaluations.policy_types import PolicyObs
from shell.linear_approximator import PRICE_COL

@dataclass
class HistoricalQuantilePolicy:
    """
    price = quantile of historical LMP conditioned on (day-of-week, hour)
    quantity = fixed quantity (MW)
    """
    action_space: ActionSpace
    lmp_df: pd.DataFrame
    quantile: float
    quantity_mw: float

    def __post_init__(self):
        if not (0.0 <= self.quantile <= 1.0):
            raise ValueError("quantile must be in (0.0, 1.0)")
        
        df = self.lmp_df.copy()
        if "datetime" not in df.columns:
            raise ValueError("lmp_df must contain a 'datetime' column (UTC timestamps)")
        
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        dt_series = pd.to_datetime(df["datetime"], utc=True)
        df["hour"] = dt_series.dt.hour
        df["dow"] = dt_series.dt.dayofweek

        if PRICE_COL not in df.columns:
            raise ValueError(f"lmp_df must contain a '{PRICE_COL}' column (LMP values)")
        
        # context -> quantile price
        self.table = (
            df.groupby(["dow", "hour"])[PRICE_COL]
              .quantile(self.quantile)
              .to_dict()
        )

        # fallback if a context is missing
        self.fallback_price = float(df[PRICE_COL].quantile(self.quantile))

    def act(self, obs: PolicyObs | None = None) -> int:
        """
        obs must contain: obs["timestamp"] (current ts), used to lookup hour/dow.
        """
        if obs is None or "timestamp" not in obs:
            raise ValueError("HistoricalQuantilePolicy requires obs['timestamp']")
        ts = pd.to_datetime(obs["timestamp"], utc=True)
        key = (int(ts.dayofweek), int(ts.hour))
        price = float(self.table.get(key, self.fallback_price))
        qty = float(self.quantity_mw)
        return self.action_space.encode_from_values(price=price, quantity=qty)