import pandas as pd
import numpy as np
import os
from typing import Dict, Optional
from linear_approximator import Discretizer

class State:
    """
    Simple time-indexed state container for a plant.

    Features:
      - plant_id
      - current step pointer
      - time-indexed raw and discretized data
      - apply discretizers
      - change_state(step)
      - get(column or timestamp)
      - append new data into the state timeline
    """

    def __init__(self, plant_id: str, discretizers: Dict[str, "Discretizer"]):
        self.plant_id = plant_id
        self.discretizers = discretizers

        self.vars: Optional[pd.DataFrame] = None  # raw merged data
        self.data: Optional[pd.DataFrame] = None  # discretized state
        self.timestamps = []
        self.ptr = 0  # current step index

    def load(self, dfs: Dict[str, pd.DataFrame], time_col="datetime"):
        """
        dfs is dict: {name: df}
        Each df must have a datetime column.
        """
        aligned = []
        for name, df in dfs.items():
            if df is None:
                continue
            if time_col not in df.columns:
                raise ValueError(f"{name} missing required time column '{time_col}'")

            temp = df.copy()
            temp[time_col] = pd.to_datetime(temp[time_col], errors="coerce")
            temp = temp.set_index(time_col)
            aligned.append(temp)

        if len(aligned) == 0:
            self.vars = pd.DataFrame()
        elif len(aligned) == 1:
            self.vars = aligned[0]
        else:
            self.vars = aligned[0].join(aligned[1:], how="outer")

        self.vars.sort_index(inplace=True)

    def apply(self):
        """
        Apply each discretizer to its configured column.
        Creates self.data: time-indexed discretized state space.
        """
        if self.vars is None:
            raise RuntimeError("Call load() first.")

        out_cols = {}

        for name, disc in self.discretizers.items():
            col = disc.col
            if col not in self.vars.columns:
                continue

            # Extract series and coerce to numeric
            series_df = self.vars[[col]].copy()
            series_df[col] = pd.to_numeric(series_df[col], errors="coerce")

            # Fit/transform
            if disc.edges_ is None:
                vals = disc.fit_transform(series_df)
            else:
                vals = disc.transform(series_df)

            out_cols[name] = pd.Series(vals, index=series_df.index)

        # final assembled state space
        self.data = pd.DataFrame(out_cols).reindex(self.vars.index)
        self.timestamps = list(self.data.index)
        self.ptr = 0

        return self.data

    def change_state(self, step_index: int):
        """
        Move the current pointer to a new timestep.
        """
        if self.data is None:
            raise RuntimeError("Call apply() first.")

        if step_index < 0 or step_index >= len(self.timestamps):
            raise IndexError("step_index out of range.")

        self.ptr = step_index

    def current_row(self) -> pd.Series:
        if self.data is None:
            raise RuntimeError("Call apply() first.")
        return self.data.iloc[self.ptr]

    def get(self, key):
        """
        If key is a column: return value at current timestamp.
        If key is a timestamp: return entire row for that timestamp.
        """
        if self.data is None:
            raise RuntimeError("Call apply() first.")

        if key in self.data.columns:
            return self.data.iloc[self.ptr][key]

        if key in self.data.index:
            return self.data.loc[key]

        raise KeyError(f"{key} is not a column or timestamp.")


    def append(self, new_df: pd.DataFrame, time_col="datetime"):
        """
        Append new rows to the raw dataset, maintaining a consistent index.
        Requires re-running apply() after append.
        """
        temp = new_df.copy()
        temp[time_col] = pd.to_datetime(temp[time_col], errors="coerce")
        temp = temp.set_index(time_col)

        if self.vars is None:
            self.vars = temp
        else:
            self.vars = pd.concat([self.vars, temp]).sort_index()


    def n_steps(self):
        return len(self.timestamps)

    def current_time(self):
        return self.timestamps[self.ptr] if self.timestamps else None

    def __repr__(self):
        return f"State(plant_id={self.plant_id}, step={self.ptr}, time={self.current_time()})"