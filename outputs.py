import pandas as pd
import numpy as np

from data_pipeline import MarketDataPipeline
from discretizer import Discretizer

MASTER_CSV = r"c:\Users\chown\Desktop\Fourth Year\Enverus Capstone\master_market_data.csv"
OUT_CSV    = r"c:\Users\chown\Desktop\Fourth Year\Enverus Capstone\state_space_inputs.csv"


def main():
    # Load merged market data
    print("[StateInputs] Loading master data from:", MASTER_CSV)
    df = pd.read_csv(MASTER_CSV)

    # Build the data pipeline (14-day history)
    print("[StateInputs] Building MarketDataPipeline...")
    pipeline = MarketDataPipeline(
        df,
        timestamp_col="timestamp",
        lmp_col="lmp",
        load_col="load_mwh",
        forecaster=None,      # Plug SARIMAX here
        history_days=14,
    )

    valid_days = pipeline.get_available_days()
    print(f"[StateInputs] Number of valid days: {len(valid_days)}")
    if not valid_days:
        raise RuntimeError("No valid days found. Check your data or pipeline settings.")

    # Fit discretizers on the full dataset (for LMP and load)
    print("[StateInputs] Fitting discretizers on full dataset...")

    lmp_disc = Discretizer(col="lmp", n_bins=8)
    lmp_disc.fit(df)

    load_disc = Discretizer(col="load_mwh", n_bins=8)
    load_disc.fit(df)

    # Loop over each valid day + each hour and build rows
    rows = []

    for day in valid_days:
        bundle = pipeline.get_day_bundle(day)
        dow = bundle["dow"]  # Zero indexed

        for hour in range(24):
            realized_lmp = float(bundle["realized_lmp"][hour])
            realized_load = float(bundle["realized_load"][hour])
            forecast_load = float(bundle["forecast_load"][hour]) if not np.isnan(bundle["forecast_load"][hour]) else np.nan

            # Discrete bins using Discretizer
            lmp_bin = lmp_disc.to_index(realized_lmp)
            load_bin = load_disc.to_index(realized_load)

            if np.isnan(forecast_load):
                forecast_bin = None
            else:
                forecast_bin = load_disc.to_index(forecast_load)

            rows.append({
                "date": day,
                "hour": hour,
                "dow": dow,
                "lmp": realized_lmp,
                "load_mwh": realized_load,
                "forecast_load": forecast_load,
                "lmp_bin": lmp_bin,
                "load_bin": load_bin,
                "forecast_bin": forecast_bin,
            })

    # Build DataFrame and save
    state_df = pd.DataFrame(rows)
    print("[StateInputs] Built state input table with rows:", len(state_df))

    state_df.to_csv(OUT_CSV, index=False)
    print("[StateInputs] Saved state-space inputs to:", OUT_CSV)


if __name__ == "__main__":
    main()
