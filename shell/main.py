import pandas as pd
import numpy as np

from market_loads_api import ISODemandController
from load_sarimax_projections import SARIMAXLoadProjections
from action_space import ActionSpace
from linear_approximator import HIST_LOAD_COL, FORECAST_LOAD_COL, LMP_CSV_PATH, PRICE_COL, Discretizer
from market_model import MarketModel, MarketParams
from state_space import State
from tabular_q_agent import TabularQLearningAgent

ISO = "ERCOT"
START_DATE = "2023-01-01"
END_DATE = "2023-01-31"

N_PRICE_BINS = 8
N_LOAD_BINS = 8
N_FORECAST_BINS = 8
N_QTY_BINS = 8
MAX_BID_QUANTITY_MW = 50

N_EPSIODES = 20
MAX_STEPS_PER_EPISODE = 24 * 7  # One week

def read_lmp_data() -> pd.DataFrame:
    '''Loads LMP data from local CSV and returns a datetime df'''

    #TODO: add ISO handling logic, currently only ERCOT

    df = pd.read_csv(LMP_CSV_PATH)
    df["datetime"] = pd.to_datetime(df["CloseDateUTC"], utc=True)
    df = df[["datetime", PRICE_COL]].sort_values("datetime")
    return df[["datetime", PRICE_COL]].sort_values("datetime")

def make_state(historic_loads: pd.DataFrame, lmp_df: pd.DataFrame) -> State:
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

def make_action_space(lmp_df: pd.DataFrame) -> ActionSpace:
    price_disc = Discretizer(col=PRICE_COL, n_bins=N_PRICE_BINS)
    price_disc.fit(lmp_df[[PRICE_COL]])

    qty_grid = np.linspace(0, MAX_BID_QUANTITY_MW, 100)
    qty_df = pd.DataFrame({ "quantity": qty_grid })
    qty_disc = Discretizer(col="quantity", n_bins=N_QTY_BINS)
    qty_disc.fit(qty_df)

    return ActionSpace(price_disc=price_disc, quantity_disc=qty_disc)

def train():
    #TODO: use Enverus API to get historic loads

    historic_load_api = ISODemandController(START_DATE, END_DATE, ISO)
    historic_loads = historic_load_api.get_market_loads()

    #TODO: forecast historic loads
    try:
        sarimax = SARIMAXLoadProjections(historic_loads)
        forecast_df = sarimax.get_forecast_df()
    except Exception as e:
        print(f"Warning: SARIMAX model failed, continuing without it: {e}")
        forecast_df = None

    lmp_df = read_lmp_data()

    state = make_state(historic_loads, lmp_df)
    action_space = make_action_space(lmp_df)

    #TODO: set market parameters appropriately, these are currently arbitrary
    market_params = MarketParams(
        marginal_cost=20.0,
        price_noise_std=5.0,
        min_price=float(action_space.price_disc.edges_[0]),
        max_price=float(action_space.price_disc.edges_[-1]),
    )
    market_model = MarketModel(action_space, market_params)

    #TODO: Explicitly pass in hyperparameters
    agent = TabularQLearningAgent(num_actions = action_space.n_actions)

    for episode in range(N_EPSIODES):
        # New episode on first observation
        observation = state.reset(new_episode=(episode > 0))
        state_key = agent.state_to_key(observation)
        total_reward = 0.0

        max_steps = min(MAX_STEPS_PER_EPISODE, state.n_steps() -1)

        for t in range(max_steps):
            action = agent.select_action(state_key)

            current_time = state.current_time()
            if state.raw_state_data is not None:
                raw_current_row = state.raw_state_data.loc[current_time]
                forecast_price = raw_current_row[PRICE_COL]

                _, _, reward = market_model.clear_market_from_action(
                    action, 
                    forecast_price
                )

                next_observation, _, done, _ = state.step(action)
                next_state_key = None if done else agent.state_to_key(next_observation)

                if next_state_key is not None:
                    agent.update_q_table(state_key, action, reward, next_state_key, done)
                    state_key = next_state_key
                    total_reward += reward

                if done:
                    break

            agent.decay_epsilon()

            print(
            f"Episode {episode + 1}/{N_EPSIODES} | "
            f"steps: {t + 1} | "
            f"total reward: {total_reward:.2f} | "
            f"epsilon: {agent.epsilon:.3f}"
        )
            
    print("Training complete.")
    return agent, state, action_space, market_model

if __name__ == "__main__":
    train()