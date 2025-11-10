import os, requests, pandas as pd
from dotenv import load_dotenv

class ISODemandController:
    ISOs = {"ERCOT":"ERCO","CAISO":"CISO","PJM":"PJM","MISO":"MISO","SPP":"SPP","NYISO":"NYIS","ISONE":"ISONE"}
    
    def __init__(self, start, end, targetISO = ""):
        load_dotenv()
        self.__api_key = os.getenv("EIA_KEY")
        self.start = start
        self.end = end
        if targetISO:
            self.ISOs = {targetISO:self.ISOs[targetISO]}

    def get_market_loads(self):
        df = pd.DataFrame()
        for iso in self.ISOs:
            r = requests.get(
                "https://api.eia.gov/v2/electricity/rto/region-data/data/",
                params={
                    "api_key": self.__api_key,
                    "frequency": "hourly",
                    "facets[respondent][]": iso,
                    "data[0]": "value",
                    "facets[type][]": "D",  # Demand (load)
                    "start": self.start, "end": self.end, "sort[0][column]": "period", "sort[0][direction]": "asc"
                },
                timeout=60
            )
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if r.status_code == 403:
                    raise RuntimeError(
                        "Received HTTP 403 Forbidden from EIA.\n"
                        "This usually means your API key is invalid, missing, or not authorized.\n\n"
                        "Check that:\n"
                        "  • You created a .env file with a valid EIA_KEY\n"
                        "  • You are loading EIA_KEY into `self.__api_key`\n"
                        "  • The key has not expired or been revoked."
                    ) from e
            df_iso = pd.DataFrame(r.json()["response"]["data"])
            df_iso["period"] = pd.to_datetime(df_iso["period"], utc=True)
            df_iso.rename(columns={"value":"load_MW"}, inplace=True)
            df_iso["respondent"] = iso
            print(len(df_iso), "rows from EIA")
            print(df_iso[["period","load_MW","respondent"]])
            df = pd.concat([df, df_iso], ignore_index=True)
        return df[["period","load_MW","respondent"]]