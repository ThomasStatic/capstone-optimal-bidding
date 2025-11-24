import pandas as pd
import numpy as np

from typing import Dict, Optional
from linear_approximator import Discretizer
from dataclasses import dataclass

@dataclass
class State:
    """State Space class for Q-Learning.

    Returns:
        _type_: Dynamic State Space object.
    """

    def __init__(self, plant_id: str, discretizers: Dict[str, "Discretizer"]):
        """

        Args:
            plant_id (str): Plant ID for state space
            discretizers (Dict[str, Discretizer]): Discretization function
        """        
        self.plant_id = plant_id
        self.plant_data: Optional[pd.DataFrame] = None  # raw plant data
        self.discretizers = discretizers

        self.state_vars: Optional[list] = None  # List of variables utilized
        self.raw_state_data: Optional[pd.DataFrame] = None  # original raw data
        self.state_data: Optional[pd.DataFrame] = None  # discretized state
        self.timestamps = []
        self.ptr = 0  # current step index

    def load_plant_data(self, df: pd.DataFrame, index_col="plant_id"):
        """Loads raw plant data.

        Args:
            df (pd.DataFrame): Raw plant data with time index.
        """
        if self.plant_id is None:
            raise RuntimeError("Plant ID not set.")
        
        if index_col not in df.columns:
            raise ValueError(f"Index column '{index_col}' not found in data.")

        if self.plant_id not in df[index_col].values:
            raise ValueError(f"Plant ID {self.plant_id} not found in data.")
        
        self.plant_data = df[df[index_col] == self.plant_id].copy()

    def load_state_data(self, dfs: Dict[str, pd.DataFrame], time_col="datetime"):
        """Generates raw state space by merging multiple dataframes on a time column.

        Args:
            dfs (Dict[str, pd.DataFrame]): Dictionary of dataframes to merge
            time_col (str, optional): Column name for time column to merge on. Defaults to "datetime".

        Raises:
            ValueError: Raises if incorrect time index provided.
        """        """
        dfs is dict: {name: df}
        Each df must have a datetime column.
        """
        aligned = []

        ## Pre-processing of dataframes
        for name, df in dfs.items():
            if df is None:
                continue
            if time_col not in df.columns:
                raise ValueError(f"{name} missing required time column '{time_col}'")

            temp = df.copy()
            temp[time_col] = pd.to_datetime(temp[time_col], errors="coerce")
            temp = temp.set_index(time_col)
            aligned.append(temp)

        ## Merge all dataframes on time index
        if len(aligned) == 0:
            self.raw_state_data = pd.DataFrame()
        elif len(aligned) == 1:
            self.raw_state_data = aligned[0]
        else:
            self.raw_state_data = aligned[0].join(aligned[1:], how="outer")

        if isinstance(self.raw_state_data, pd.DataFrame):
            self.raw_state_data.sort_index(inplace=True)

        self.vars = [c for c in self.raw_state_data.columns if c != time_col]


    def apply(self):
        """Applies Descriptizers to raw_state_data.

        Raises:
            RuntimeError: Raw State data not loaded.

        Returns:
            _type_: Dataframe with discretized state space.
        """        """
        Apply each discretizer to its configured column.
        Creates self.data: time-indexed discretized state space.
        """
        if self.raw_state_data is None or not isinstance(self.raw_state_data, pd.DataFrame):
            raise RuntimeError("No raw_state_data available. Call load_state_data() first.")

        vars_list = self.vars

        out_cols = {}

        for col in vars_list:
            # Work with a one-column DataFrame for compatibility
            series_df = self.raw_state_data[[col]].copy()

            # Try coercing to numeric if possible
            series_df[col] = (
                pd.to_numeric(series_df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )


            # Find provided discretizer (match on d.col)
            disc = None
            for d in self.discretizers.values():
                if getattr(d, "col", None) == col:
                    disc = d
                    break

            # If none exists, use default
            if disc is None:
                disc = Discretizer(col=col)

            # Fit/transform
            if getattr(disc, "edges_", None) is None:
                vals = disc.fit_transform(series_df)
            else:
                vals = disc.transform(series_df)

            out_cols[col] = pd.Series(vals, index=series_df.index, name=col)

        # --- Time-of-day feature ---
        hours = pd.Series(self.raw_state_data.index.hour, index=self.raw_state_data.index)

        def map_hour_to_period(h):
            if 0 <= h <= 5:
                return 0
            if 7 <= h <= 10:
                return 1
            if 11 <= h <= 15:
                return 2
            if 15 <= h <= 20:
                return 3
            if 21 <= h <= 23:
                return 4
            return 5

        time_of_day = hours.map(map_hour_to_period).astype(int)
        out_cols["time_of_day"] = pd.Series(time_of_day, index=self.raw_state_data.index)

        # Final dataframe
        ordered_cols = vars_list + ["time_of_day"]
        self.state_vars = ordered_cols
        self.state_data = pd.DataFrame(out_cols).reindex(self.raw_state_data.index)[ordered_cols]

        self.timestamps = list(self.state_data.index)
        self.ptr = 0

        return self.state_data

    def reset(self):
        """Reset environment to starting state.

        Raises:
            RuntimeError: Discretized state data not loaded.

        Returns:
            _type_: Current state as numpy array.
        """
        if self.state_data is None:
            raise RuntimeError("Call apply() before reset().")

        self.ptr = 0
        return self.current_row().values

    def step(self, action=None):
        """
        Take one environment step forward.
        Action is unused for now (exogenous environment).
        """
        if self.state_data is None:
            raise RuntimeError("Call apply() before step().")

        self.ptr += 1

        if self.ptr >= len(self.state_data):
            self.ptr = len(self.state_data) - 1
            done = True
        else:
            done = False

        next_state = self.current_row().values

        # placeholder exogenous reward
        reward = 0.0

        info = {
            "timestamp": self.current_time(),
            "step": self.ptr
        }

        return next_state, reward, done, info


    def change_state(self, step_index: int):
        """
        Move the current pointer to a new timestep.
        """
        if self.state_data is None:
            raise RuntimeError("Call apply() first.")

        if step_index < 0 or step_index >= len(self.timestamps):
            raise IndexError("step_index out of range.")

        self.ptr = step_index

    def current_row(self) -> pd.Series:
        if self.state_data is None:
            raise RuntimeError("Call apply() first.")
        return self.state_data.iloc[self.ptr]

    def get(self, key):
        """
        If key is a column: return value at current timestamp.
        If key is a timestamp: return entire row for that timestamp.
        """
        if self.state_data is None:
            raise RuntimeError("Call apply() first.")

        if key in self.state_data.columns:
            return self.state_data.iloc[self.ptr][key]

        if key in self.state_data.index:
            return self.state_data.loc[key]

        raise KeyError(f"{key} is not a column or timestamp.")


    def append(self, new_df: pd.DataFrame, time_col="datetime"):
        """
        Append new rows to the raw dataset, maintaining a consistent index.
        Requires re-running apply() after append.
        """
        temp = new_df.copy()
        temp[time_col] = pd.to_datetime(temp[time_col], errors="coerce")
        temp = temp.set_index(time_col)

        if self.raw_state_data is None:
            self.raw_state_data = temp
        else:
            self.raw_state_data = pd.concat([self.raw_state_data, temp]).sort_index()


    def n_steps(self):
        return len(self.timestamps)

    def current_time(self):
        return self.timestamps[self.ptr] if self.timestamps else None

    def __repr__(self):
        return f"State(plant_id={self.plant_id}, step={self.ptr}, time={self.current_time()})"

## Test on Dummy Data
if __name__ == "__main__":

    discretizers = {
        "load": Discretizer(col="load"),
        "price": Discretizer(col="price"),
        "gas": Discretizer(col="gas_index")
    }

    state = State(plant_id=2343, discretizers=discretizers)

    # --- Two years of hourly data ---
    start = pd.Timestamp("2023-01-01 00:00:00")
    periods = 2 * 365 * 24
    idx = pd.date_range(start=start, periods=periods, freq="h")

    rng = np.random.default_rng(seed=42)

    daily = 100 + 30 * np.sin(2 * np.pi * (idx.hour / 24))
    yearly = 20 * np.sin(2 * np.pi * (idx.dayofyear / 365))
    weekly = 10 * np.where(idx.dayofweek < 5, 1.0, 0.8)

    load = (daily + yearly) * weekly + rng.normal(scale=5, size=len(idx))
    price = 30 + 0.2 * load + rng.normal(0, 3, len(idx))
    gas = 2 + 0.01 * load + rng.normal(0, 0.05, len(idx))

    df_load = pd.DataFrame({"datetime": idx, "load": load})
    df_price = pd.DataFrame({"datetime": idx, "price": price})
    df_gas = pd.DataFrame({"datetime": idx, "gas_index": gas})

    state.load_state_data({
        "load": df_load,
        "price": df_price,
        "gas": df_gas
    })

    state.apply()

    # --- RL interaction example ---
    obs = state.reset()
    print("Initial state:", obs)

    done = False
    step_counter = 0

    while not done and step_counter < 10:
        action = None  # placeholder
        obs, reward, done, info = state.step(action)

        print(f"\nStep {step_counter+1}")
        print("State:", obs)
        print("Time:", info["timestamp"])

        step_counter += 1
