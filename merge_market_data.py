import pandas as pd

LMP_FILE = r"c:/Users\chown\Desktop\Fourth Year\Enverus Capstone\lmp-hourly.csv"
LOAD_FILE = r"c:/Users\chown\Downloads\capstone-optimal-bidding\hourly_load.csv"

def main():
    # Load LMP data
    print("[Merge] Loading LMP data from:", LMP_FILE)
    lmp_df = pd.read_csv(LMP_FILE)

    print("[Merge] LMP columns:", lmp_df.columns.tolist())

    if "CloseDateUTC" not in lmp_df.columns or "lmpPrice" not in lmp_df.columns:
        raise KeyError(
            "Expected columns 'CloseDateUTC' and 'lmpPrice' in LMP file. "
            f"Found: {lmp_df.columns.tolist()}"
        )

    lmp_df = lmp_df[["CloseDateUTC", "lmpPrice"]].rename(
        columns={"CloseDateUTC": "timestamp", "lmpPrice": "lmp"}
    )

    lmp_df["timestamp"] = pd.to_datetime(lmp_df["timestamp"])
    lmp_df = lmp_df.sort_values("timestamp").reset_index(drop=True)

    print("[Merge] LMP rows:", len(lmp_df))
    print("  LMP from:", lmp_df["timestamp"].min())
    print("  LMP to:  ", lmp_df["timestamp"].max())

    # Load load data
    print("\n[Merge] Loading LOAD data from:", LOAD_FILE)
    load_df = pd.read_csv(LOAD_FILE)

    print("[Merge] LOAD columns before rename:", load_df.columns.tolist())

    cols = load_df.columns.tolist()

    if "timestamp" not in cols:
        if "period" in cols:
            load_df = load_df.rename(columns={"period": "timestamp"})
        else:
            raise KeyError(
                "Could not find 'timestamp' or 'period' column in LOAD file. "
                f"Found: {cols}"
            )

    if "load_MW" not in cols:
        if "value" in cols:
            load_df = load_df.rename(columns={"value": "load_MW"})
        elif "load_mwh" in cols:
            load_df = load_df.rename(columns={"load_mwh": "load_MW"})

    if "load_MW" not in load_df.columns:
        raise KeyError(
            "Could not find a load column ('load_MW' or 'value') in LOAD file. "
            f"Found: {load_df.columns.tolist()}"
        )

    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"])
    load_df = load_df.sort_values("timestamp").reset_index(drop=True)

    print("[Merge] LOAD rows:", len(load_df))
    print("  LOAD from:", load_df["timestamp"].min())
    print("  LOAD to:  ", load_df["timestamp"].max())

    # Merge LMP + LOAD on timestamp
    print("\n[Merge] Merging LMP + LOAD on timestamp using merge_asof...")

    master_df = pd.merge_asof(
        lmp_df.sort_values("timestamp"),
        load_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    )

    # Rename load column to load_mwh for pipeline use
    if "load_MW" in master_df.columns:
        master_df = master_df.rename(columns={"load_MW": "load_mwh"})

    print("[Merge] Master rows:", len(master_df))
    print("  Master from:", master_df["timestamp"].min())
    print("  Master to:  ", master_df["timestamp"].max())
    print("  Master columns:", master_df.columns.tolist())

    out_path = r"c:\Users\chown\Desktop\Fourth Year\Enverus Capstone\master_market_data.csv"
    master_df.to_csv(out_path, index=False)
    print(f"\n[Merge] Saved merged data to: {out_path}")


if __name__ == "__main__":
    main()
