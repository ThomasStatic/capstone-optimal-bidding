import pandas as pd
import numpy as np
from datetime import datetime, date


class MarketDataPipeline:

    def __init__(
        self,
        df,
        timestamp_col="timestamp",
        lmp_col="lmp",
        load_col="load_mwh",
        forecaster=None,
        history_days=14,
    ):
        
        self.timestamp_col = timestamp_col
        self.lmp_col = lmp_col
        self.load_col = load_col
        self.forecaster = forecaster
        self.history_days = history_days

        df = df.copy()
        self._prepare_base_df(df)
        self._compute_valid_days()


    # Internal helpers for cleaning and day selection
    def _prepare_base_df(self, df):

        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # Sort by time
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        # Drop rows with missing essential values
        df = df.dropna(subset=[self.timestamp_col, self.lmp_col, self.load_col])

        # Create date and hour columns
        df["date"] = df[self.timestamp_col].dt.date
        df["hour"] = df[self.timestamp_col].dt.hour
        df["dow"] = df[self.timestamp_col].dt.dayofweek  # zero indexed


        agg_df = (
            df.groupby(["date", "hour"], as_index=False)
              .agg({
                  self.timestamp_col: "min",   
                  self.lmp_col: "mean",       
                  self.load_col: "mean", 
              })
        )

        # Recompute day-of-week
        agg_df["dow"] = agg_df[self.timestamp_col].dt.dayofweek

        # Cleaned df
        self.df = agg_df

        # Unique sorted list of dates in the dataset
        self.dates = sorted(self.df["date"].unique().tolist())

    def _compute_valid_days(self):
        """
        - Have exactly 24 hours of data (0..23)
        - Have 'history_days' days of data before them
        """
        valid_days = []

        hours_per_day = (
            self.df.groupby("date")["hour"]
            .nunique()
            .to_dict()
        )

        for idx, d in enumerate(self.dates):

            if hours_per_day.get(d, 0) != 24:
                continue

            if idx < self.history_days:
                continue

            history_ok = True
            for offset in range(1, self.history_days + 1):
                prev_day = self.dates[idx - offset]
                if hours_per_day.get(prev_day, 0) != 24:
                    history_ok = False
                    break

            if history_ok:
                valid_days.append(d)

        self.valid_days = valid_days


    def get_available_days(self):

        return self.valid_days


    def has_day(self, day):

        day = self._normalize_date(day)
        return day in self.valid_days


    def get_day_bundle(self, day):
        """
        Returns a dict with:
          - 'date' : datetime.date
          - 'dow'  : int (0..6)
          - 'lmp_history'   : np.ndarray shape (24, history_days)
          - 'load_history'  : np.ndarray shape (24, history_days)
          - 'realized_lmp'  : np.ndarray shape (24,)
          - 'realized_load' : np.ndarray shape (24,)
          - 'forecast_load' : np.ndarray shape (24,) or None
        """
        day = self._normalize_date(day)

        if day not in self.valid_days:
            raise ValueError(
                f"Day {day} is not in valid_days. "
                f"Call get_available_days() to see valid options."
            )

        day_idx = self.dates.index(day)

        day_mask = self.df["date"] == day
        day_df = self.df[day_mask].sort_values("hour")

        if day_df["hour"].nunique() != 24:
            raise RuntimeError(
                f"Day {day} does not actually have 24 unique hours after cleaning."
            )

        # Realized LMP and load for this day
        realized_lmp = day_df[self.lmp_col].to_numpy()
        realized_load = day_df[self.load_col].to_numpy()
        dow = int(day_df["dow"].iloc[0])

        # Build 14-day history matrices for each hour
        lmp_history = np.zeros((24, self.history_days), dtype=float)
        load_history = np.zeros((24, self.history_days), dtype=float)


        for h in range(24):
            for k in range(self.history_days):
                offset = self.history_days - k  # So k=0 is the oldest day
                prev_day = self.dates[day_idx - offset]

                val_lmp = self._get_value_for_day_hour(prev_day, h, self.lmp_col)
                val_load = self._get_value_for_day_hour(prev_day, h, self.load_col)

                lmp_history[h, k] = val_lmp
                load_history[h, k] = val_load


        forecast_load = self._get_forecast_for_day(day_df)

        bundle = {
            "date": day,
            "dow": dow,
            "lmp_history": lmp_history,
            "load_history": load_history,
            "realized_lmp": realized_lmp,
            "realized_load": realized_load,
            "forecast_load": forecast_load,
        }

        return bundle

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalize_date(self, d):
        """
        Convert various date formats to datetime.date.
        Accepts:
          - datetime.date
          - datetime.datetime
          - 'YYYY-MM-DD' string
        """
        if isinstance(d, date) and not isinstance(d, datetime):
            return d
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, str):
            return pd.to_datetime(d).date()
        raise TypeError(f"Unsupported date type: {type(d)}")


    def _get_value_for_day_hour(self, day, hour, col):
        """
        Return the scalar value in column 'col' for a given date + hour.
        Assumes data has been cleaned such that this exists.
        """
        mask = (self.df["date"] == day) & (self.df["hour"] == hour)
        subset = self.df.loc[mask, col]

        if subset.empty:
            raise RuntimeError(f"No data for {day} at hour {hour} in column {col}.")

        return float(subset.iloc[0])


    def _get_forecast_for_day(self, day_df):
        """
        The forecaster can implement either:
          - get_24_hour_forecast(start_timestamp)
          - forecast_day(start_timestamp)

        If no forecaster is attached, returns a vector of NaNs.
        """
        # Starting timestamp of the day (hour 0)
        start_ts = day_df[self.timestamp_col].min()

        if self.forecaster is None:
            # Return NaNs if no forecast model is attached
            return np.full(24, np.nan, dtype=float)

        # Try common method names and adapt as needed to your actual class
        if hasattr(self.forecaster, "get_24_hour_forecast"):
            forecast = self.forecaster.get_24_hour_forecast(start_ts)
        elif hasattr(self.forecaster, "forecast_day"):
            forecast = self.forecaster.forecast_day(start_ts)
        else:
            raise AttributeError(
                "Forecaster object must implement 'get_24_hour_forecast' "
                "or 'forecast_day(start_timestamp)'."
            )

        forecast = np.asarray(forecast, dtype=float)

        if forecast.shape[0] != 24:
            raise RuntimeError(
                f"Expected 24-hour forecast, got shape {forecast.shape}."
            )

        return forecast