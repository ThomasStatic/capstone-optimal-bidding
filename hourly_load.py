import requests
import pandas as pd

API_KEY = "ggWXKFf7JGz53AJgpb7uGZk9msQvvbbQd9NcVdDF"


def get_hourly_load(start, end, respondent):

    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

    start_period = start + "T00"
    end_period = end + "T23"

    all_rows = []
    offset = 0
    length = 5000

    print(f"[EIA] Requesting hourly load for {respondent} from {start} to {end}")

    while True:
        params = {
            "api_key": API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": respondent,
            "facets[type][]": "D",
            "start": start_period,
            "end": end_period,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": length,
        }

        resp = requests.get(url, params=params)


        if resp.status_code != 200:
            print("\n[ERROR] HTTP", resp.status_code)
            print("Response text:")
            print(resp.text)
            return None

        data = resp.json()


        if "response" not in data:
            print("\n[ERROR] No 'response' field in JSON.")
            print("Full JSON:")
            print(data)
            return None

        chunk = data["response"].get("data", [])

        if not chunk:
            break

        all_rows.extend(chunk)


        if len(chunk) < length:
            break

        offset += length

    if not all_rows:
        print("\n[ERROR] No data rows returned.")
        return None

    df = pd.DataFrame(all_rows)


    if not {"period", "value"}.issubset(df.columns):
        print("\n[ERROR] 'period' or 'value' columns missing in data.")
        print("Columns present:", df.columns.tolist())
        return None

    df = df[["period", "value", "respondent"]].copy()
    df = df.rename(columns={"period": "timestamp", "value": "load_MW"})

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"[EIA] Retrieved {len(df)} rows.")
    print("  From:", df['timestamp'].min())
    print("  To:  ", df['timestamp'].max())

    return df


def main():

    start_date = "2024-11-24" 
    end_date   = "2025-11-24" 
    respondent = "ERCO"

    if API_KEY == "YOUR_EIA_API_KEY_HERE":
        print("[ERROR] You forgot to paste your EIA API key into API_KEY.")
        return

    df = get_hourly_load(start_date, end_date, respondent)
    if df is None:
        print("\n[EIA] Failed to fetch data.")
        return

    # Save to CSV file
    out_path = "hourly_load.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[EIA] Saved hourly load to {out_path}")


if __name__ == "__main__":
    main()
