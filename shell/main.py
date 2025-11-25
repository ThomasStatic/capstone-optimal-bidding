import pandas as pd

from market_loads_api import ISODemandController
from load_sarimax_projections import SARIMAXLoadProjections
from shell.linear_approximator import HIST_LOAD_COL, LMP_CSV_PATH, PRICE_COL, Discretizer
from shell.state_space import State

ISO = "ERCOT"
START_DATE = "2023-01-01"
END_DATE = "2023-01-31"

N_PRICE_BINS = 8
N_LOAD_BINS = 8
MAX_BID_QUANTITY_MW = 50

N_EPSIODES = 20
MAX_STEPS_PER_EPISODE = 24 * 7  # One week

def read_lmp_data():
    '''Loads LMP data from local CSV and returns a datetime df'''

    #TODO: add error handling for date time index

    df = pd.read_csv(LMP_CSV_PATH)
    return df[["datetime", PRICE_COL]].sort_values("datetime")

def make_state(historic_loads: pd.DataFrame, lmp_df: pd.DataFrame) -> State:
    '''Build the state space, discretized'''

    df_load = (
        historic_loads.rename(columns={"period": "datetime"})[
            ["datetime", HIST_LOAD_COL]
        ]
        .copy()
    )
    df_load["datetime"] = pd.to_datetime(df_load["datetime"], utc=True)

    #TODO: add SARIMAX forecast data
    dfs = {
        "load": df_load,
        "price": lmp_df,
    }

    #TODO: add SARIMXAX discretization
    state_discretizers = {
        HIST_LOAD_COL: Discretizer(col=HIST_LOAD_COL, n_bins=N_LOAD_BINS),
        PRICE_COL: Discretizer(col=PRICE_COL, n_bins=N_PRICE_BINS),
    }

    #TODO: plant_id should work on a dict of ISOs instead of one
    state = State(plant_id=ISO, discretizers=state_discretizers)
    state.load_state_data(dfs, time_col="datetime")
    state.apply()
    return state


