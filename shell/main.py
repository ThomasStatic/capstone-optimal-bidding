import argparse
import pandas as pd
import numpy as np
import pickle

from shell.market_loads_api import ISODemandController
from shell.load_sarimax_projections import SARIMAXLoadProjections
from shell.action_space import ActionSpace
from shell.linear_approximator import (
    HIST_LOAD_COL,
    FORECAST_LOAD_COL,
    LMP_CSV_PATH,
    PRICE_COL,
    Discretizer,
)
from shell.market_model import MarketModel, MarketParams
from shell.baselines.cost_plus_markup import CostPlusMarkupPolicy
from shell.baselines.historical_quantile import HistoricalQuantilePolicy
from shell.evaluations.baseline_runner import run_policy_on_episodes
from shell.evaluations.policy_types import Policy
from shell.state_space import State
from shell.tabular_q_agent import TabularQLearningAgent

ISO = "ERCOT"
START_DATE = "2023-01-01"
END_DATE = "2023-01-31"

NUM_DISCRETIZER_BINS = 8
MAX_BID_QUANTITY_MW = 50

def build_state_and_discretizers(
    load_df: pd.DataFrame, price_df: pd.DataFrame
) -> tuple[State, dict]:
    load_disc = Discretizer(col=HIST_LOAD_COL, n_bins=NUM_DISCRETIZER_BINS)
    forecast_disc = Discretizer(col=FORECAST_LOAD_COL, n_bins=NUM_DISCRETIZER_BINS)
    price_disc = Discretizer(col=PRICE_COL, n_bins=NUM_DISCRETIZER_BINS)

    # plant id can be any string identifier - not currently used in logic
    state = State(
        plant_id="plant_1",
        discretizers={
            "load": load_disc,
            "forecast": forecast_disc,
            "price": price_disc,
        },
    )

    state.load_state_data(
        {
            "load": load_df.rename(columns={"period": "datetime"}),
            "price": price_df,
        }
    )

    # On first pass, discretize without any forecast data
    state.apply()

    return state, {
        "load": load_disc,
        "forecast": forecast_disc,
        "price": price_disc,
    }

# When we call subsequent state.apply() during episodes, we will have forecast data
def inject_epsisode_forecast(state: State, forecast_df: pd.DataFrame) -> None:
    '''Injects 24-hour forecast data into the state for the episode'''
    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been intialized")
    
    df = state.raw_state_data

    if FORECAST_LOAD_COL not in df.columns:
        df[FORECAST_LOAD_COL] = np.nan

    forecast_df = forecast_df.copy()
    forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"], utc=True)

    f = forecast_df.set_index("datetime")[FORECAST_LOAD_COL]
    df.loc[f.index, FORECAST_LOAD_COL] = f.values

def make_action_space(lmp_df: pd.DataFrame) -> ActionSpace:
    price_disc = Discretizer(col=PRICE_COL, n_bins=NUM_DISCRETIZER_BINS)
    price_disc.fit(lmp_df[[PRICE_COL]])

    qty_grid = np.linspace(0, MAX_BID_QUANTITY_MW, 100)
    qty_df = pd.DataFrame({ "quantity": qty_grid })
    qty_disc = Discretizer(col="quantity", n_bins=NUM_DISCRETIZER_BINS)
    qty_disc.fit(qty_df)

    return ActionSpace(price_disc=price_disc, quantity_disc=qty_disc)

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    historic_load_api = ISODemandController(START_DATE, END_DATE, ISO)
    load_df = historic_load_api.get_market_loads()
    load_df =load_df[load_df["respondent"] == ISO].copy()
    load_df["period"] = pd.to_datetime(load_df["period"], utc=True)
    load_df = load_df.sort_values("period")

    lmp_df = pd.read_csv(LMP_CSV_PATH, parse_dates=["CloseDateUTC"])
    lmp_df = lmp_df.rename(columns={"CloseDateUTC": "datetime"})
    lmp_df["datetime"] = pd.to_datetime(lmp_df["datetime"], utc=True)
    lmp_df = lmp_df.sort_values("datetime")

    return load_df, lmp_df

def build_world(load_df: pd.DataFrame, lmp_df: pd.DataFrame) -> tuple[State, ActionSpace, MarketModel]:
    state, _ = build_state_and_discretizers(load_df, lmp_df)
    action_space = make_action_space(lmp_df)

    # TODO: Come up with better market params
    market_params = MarketParams(
        marginal_cost=20.0, 
        price_noise_std=5.0,
        min_price=float(action_space.price_disc.edges_[0]),
        max_price=float(action_space.price_disc.edges_[-1]),
    )
    market_model = MarketModel(action_space, market_params)
    return state, action_space, market_model


def train(n_epsiodes = 20) -> tuple[TabularQLearningAgent, State, ActionSpace, MarketModel]:
    load_df, lmp_df = load_data()

    state, action_space, market_model = build_world(load_df, lmp_df)

    agent = TabularQLearningAgent(num_actions=action_space.n_actions)

    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been intialized")
    
    episode_starts = list(range(0, len(state.raw_state_data) - state.window_size + 1, state.step_hours))
    episode_starts = episode_starts[:n_epsiodes] # Limit to n_episodes

    for ep_idx, start_idx in enumerate(episode_starts):
        print(f"\n=== EPISODE {ep_idx+1}/{len(episode_starts)} | start_idx={start_idx} ===")
        cumulative_reward = 0

        state.episode_start = start_idx
        episode_start_ts = state.raw_state_data.index[start_idx]

        # Build history for SARIMAX only using data prior to episode start
        history_mask = load_df["period"] < episode_start_ts
        history_for_model = load_df.loc[history_mask].copy()
        if history_for_model.empty:
            # Minimum history requirement for start 
            history_for_model = load_df.iloc[:state.window_size].copy()
        sarimax = SARIMAXLoadProjections(history_for_model)
        forecast_df = sarimax.get_forecast_df()

        inject_epsisode_forecast(state, forecast_df)

        # Re-discretize now that we have forecast data
        state.apply()
        obs = state.reset(new_episode=False)
        state_key = agent.state_to_key(obs)

        done = False
        step_counter = 0

        while not done:
            #TODO: Experiment with temperature parameter
            action_idx = agent.select_softmax_action(state_key, temperature=1.0)

            ts = state.timestamps[state.ptr]

            # Bidding 24 hours in advance
            delivery_time = ts + pd.Timedelta(hours=24)

            if delivery_time not in state.raw_state_data.index:
                # If we don't have data for the delivery time, end episode
                print(f"  No data for delivery time {delivery_time}, ending episode.")
                done = True
                break

            delivery_row = state.raw_state_data.loc[delivery_time]
            price_val = delivery_row.get(PRICE_COL, np.nan)
            if pd.isna(price_val):
                price_val = float(lmp_df[PRICE_COL].iloc[-1]) # Fallback to last known price
        
            _, _, reward = market_model.clear_market_from_action(action_idx, price_val)

            next_obs, _, done, info = state.step()

            # --- LOGGING ---
            if step_counter == 0:
                cumulative_reward = 0  # initialize at episode start

            cumulative_reward += reward

            print(
                f"[EP {ep_idx+1} | Step {step_counter}] "
                f"ts={ts} | "
                f"state_key={state_key} | "
                f"action={action_idx} | "
                f"price={price_val:.2f} | "
                f"reward={reward:.4f} | "
                f"cumulative_reward={cumulative_reward:.4f}"
            )

            next_state_key = agent.state_to_key(next_obs)

            agent.update_q_table(
                state_key,
                action_idx,
                reward,
                next_state_key,
                done
            )

            state_key = next_state_key
            obs = next_obs
            step_counter += 1

            if step_counter >= state.window_size:
                done = True

        print(f"Episode {ep_idx+1} finished after {step_counter} steps")
         
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.Q, f)

    with open("policy.pkl", "wb") as f:
        pickle.dump(agent.extract_softmax_policy(), f)
    
    print("Training complete. Q-table and policy saved.")
    return agent, state, action_space, market_model

def run_baselines(
        *, # Keyward-only arguments since complex signature
        baseline: str,
        n_episodes: int,
        markup: float,
        quantile:float,
):
    load_df, lmp_df = load_data()
    state, action_space, market_model = build_world(load_df, lmp_df)  
    
    policy: Policy
    if baseline == "cost_plus":
        policy = CostPlusMarkupPolicy(
            action_space = action_space,
            marginal_cost = 20.0,
            markup = markup,
            quantity_mw = MAX_BID_QUANTITY_MW
        )
    elif baseline == "hist_quantile":
        policy = HistoricalQuantilePolicy(
            action_space = action_space,
            lmp_df= lmp_df,
            quantile = quantile,
            quantity_mw = MAX_BID_QUANTITY_MW
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    df = run_policy_on_episodes(
        policy=policy,
        state=state,
        market_model=market_model,
        load_df=load_df,
        lmp_df=lmp_df,
        n_episodes=n_episodes,
        inject_forecast_fn=inject_epsisode_forecast,
        verbose = args.verbose
    )

def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--mode", choices=["train", "baseline"], default="train")

    p.add_argument("--n_episodes", type=int, default=20)

    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # baseline specific
    p.add_argument("--baseline", choices=["cost_plus", "hist_quantile"], default="cost_plus")
    p.add_argument("--markup", type=float, default=10.0, help="Markup for cost_plus baseline")
    p.add_argument("--quantile", type=float, default=0.7, help="Quantile for hist_quantile baseline")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(n_epsiodes=args.n_episodes)
    elif args.mode == "baseline":
        run_baselines(
            baseline=args.baseline,
            n_episodes=args.n_episodes,
            markup=args.markup,
            quantile=args.quantile
        )