import argparse
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import csv
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

def infer_market_params_from_lmp(
    lmp_df: pd.DataFrame,
    price_col: str,
    *, # Keyword-only arguments
    mc_tail_q: float = 0.25, # use lower 25% of lmp (i.e. when market is slack)
    vol_clip_q: float = 0.995, # clip extreme prices
) -> tuple[float, float]:
    '''Infer marginal cost from historical LMP data'''
    price_series = pd.to_numeric(lmp_df[price_col], errors="coerce").dropna()

    cutoff_price = price_series.quantile(mc_tail_q)
    marginal_cost = price_series[price_series <= cutoff_price].mean()

    diffs = price_series.diff().dropna()
    cap = diffs.abs().quantile(vol_clip_q)
    diffs = diffs.clip(-cap, cap)
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    price_noise_std = mad * 1.4826  # converts MAD to std under normality assumption

    if not np.isfinite(marginal_cost):
        marginal_cost = float(price_series.quantile(0.10))
    if not np.isfinite(price_noise_std):
        price_noise_std = float(diffs.std(ddof=0)) if len(diffs) else 0.0

    return marginal_cost, price_noise_std

def build_world(load_df: pd.DataFrame, lmp_df: pd.DataFrame) -> tuple[State, ActionSpace, MarketModel]:
    state, _ = build_state_and_discretizers(load_df, lmp_df)
    action_space = make_action_space(lmp_df)

    price_edges = action_space.price_disc.edges_
    if price_edges is None:
        raise RuntimeError("price_disc.edges_ is None; call fit(...) first.")
    
    marginal_cost, price_noise_std = infer_market_params_from_lmp(lmp_df, PRICE_COL)
    market_params = MarketParams(
        marginal_cost=marginal_cost, 
        price_noise_std=price_noise_std,
        min_price=float(price_edges[0]),
        max_price=float(price_edges[-1]),
    )
    market_model = MarketModel(action_space, market_params)
    return state, action_space, market_model


def train(n_epsiodes = 20) -> tuple[TabularQLearningAgent, State, ActionSpace, MarketModel]:
    load_df, lmp_df = load_data()

    state, action_space, market_model = build_world(load_df, lmp_df)

    price_q = float(lmp_df[PRICE_COL].abs().quantile(args.max_notional_q))
    max_notional = float(MAX_BID_QUANTITY_MW * price_q)

    if(args.verbose):
        print(f"[Risk] max_notional_q={args.max_notional_q} | price_q={price_q:.4f} | max_notional={max_notional:.4f}")
    
    episode_profits = np.ndarray([])
    episode_variances = np.ndarray([])
    episode_variance_per_profits =  np.ndarray([])
    episode_metrics = []
    train_episodes = 52


    agent = TabularQLearningAgent(num_actions=action_space.n_actions)

    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been intialized")
    
    episode_starts = list(range(0, len(state.raw_state_data) - state.window_size + 1, state.step_hours))
    episode_starts = episode_starts[:n_epsiodes] # Limit to n_episodes

    for ep_idx, start_idx in enumerate(episode_starts):
        print(f"\n=== EPISODE {ep_idx+1}/{len(episode_starts)} | start_idx={start_idx} ===")
        cumulative_reward = np.ndarray([])

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
            if ep_idx < train_episodes:
                action_idx_raw = agent.select_softmax_action(state_key, temperature=1.0)
            else:
                action_idx_raw = agent.select_softmax_action(state_key)

            action_idx, clip_info = action_space.project_to_feasible(
                action_idx_raw,
                max_quantity=MAX_BID_QUANTITY_MW,
                max_notional=max_notional,
            )

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

            if args.risk_penalty_lambda > 0.0 and bool(clip_info["clipped"]):
                # relative penalty based on how far over notional raw action was
                orig_notional = float(clip_info["original_notional"])
                if max_notional > 1e-12:
                    severity = max(0.0, (orig_notional - max_notional) / max_notional)
                    if args.verbose: 
                        print(f"[Risk Penalty] step={step_counter} | orig_notional={orig_notional:.4f} | max_notional={max_notional:.4f} | severity={severity:.4f}")
                        
                else:
                    severity = 0.0
                penalty = float(args.risk_penalty_lambda * (1.0 + severity))
                reward -= penalty

            next_obs, _, done, info = state.step()

            # --- LOGGING ---
            if step_counter == 0:
                cumulative_reward = np.ndarray([])  # initialize at episode start

            cumulative_reward = np.append(cumulative_reward, reward)

            print(
                f"[EP {ep_idx+1} | Step {step_counter}] "
                f"ts={ts} | "
                f"state_key={state_key} | "
                f"action={action_idx} | "
                f"price={price_val:.2f} | "
                f"reward={reward:.4f} | "
                f"cumulative_reward={np.sum(cumulative_reward):.4f}"
            )

            next_state_key = agent.state_to_key(next_obs)

            ## Freeze learning past the burn-in period
            if ep_idx < train_episodes:
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
    
        if (args.verbose):
            ## For plotting of profit
            episode_profit = cumulative_reward.sum()
            episode_variance = cumulative_reward.var(ddof=1)
            episode_variance_per_profit = episode_variance / (abs(episode_profit) + 1e-8)
            episode_profits = np.append(episode_profits, episode_profit)
            episode_variances = np.append(episode_variances, episode_variance)
            episode_variance_per_profits = np.append(episode_variance_per_profits, episode_variance_per_profit)
            episode_metrics.append({
                "episode": ep_idx + 1,
                "profit": episode_profit,
                "profit_variance": episode_variance,
                "risk_adjusted_variance": episode_variance_per_profit,
                "steps": step_counter,
            }) 
    
    if args.verbose:
        ## Note that denote the plot pre and post burn in
        episodes = np.arange(1, len(episode_profits) + 1)
        # Create output directory
        out_dir = Path("Analysis/Metrics/plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        todaysdate = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

        ## Profit
        plt.plot(episodes, episode_profits, label="Episode Profit")
        plt.axvspan(
            1,
            train_episodes,
            alpha=0.15,
            label="Burn-in Period"
        )
        plt.title("Episode Profits (Pre/Post Training)")
        plt.xlabel("Episode")
        plt.ylabel("Profit")
        plt.legend()
        plt.tight_layout()

        # Save figure
        save_path = out_dir / f"episode_profits_{todaysdate}_{train_episodes}_epi.png"
        plt.savefig(save_path)
        plt.close()

        plt.plot(episodes, episode_variances, label="Profit Variance")
        plt.axvspan(
            1,
            train_episodes,
            alpha=0.15,
            label="Burn-in Period"
        )
        plt.title("Intra-episode Profit Variance")
        plt.xlabel("Episode")
        plt.ylabel("Variance")
        plt.legend()
        plt.tight_layout()

        # Save figure
        save_path = out_dir / f"episode_variances_{todaysdate}_{train_episodes}_epi.png"
        plt.savefig(save_path)
        plt.close()

        plt.plot(episodes, episode_variance_per_profits, label="Variance / |Profit|")
        plt.axvspan(
            1,
            train_episodes,
            alpha=0.15,
            label="Burn-in Period"
        )
        plt.title("Risk-Adjusted Variance per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Variance / |Profit|")
        plt.legend()
        plt.tight_layout()

        # Save figure
        save_path = out_dir / f"profit_adjusted_episode_variances_{todaysdate}_{train_episodes}_epi.png"
        plt.savefig(save_path)
        plt.close()

        out_path = out_dir / f"episode_metrics_{todaysdate}_{train_episodes}_epi.csv"

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=episode_metrics[0].keys()
            )
            writer.writeheader()
            writer.writerows(episode_metrics)

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
    
    marginal_cost, _ = infer_market_params_from_lmp(lmp_df, PRICE_COL)

    policy: Policy
    if baseline == "cost_plus":
        policy = CostPlusMarkupPolicy(
            action_space = action_space,
            marginal_cost = marginal_cost,
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

    # risk constraints
    p.add_argument("--max_notional_q", type=float, default=0.95, help="Quantile of |LMP| used to set max_notional")
    p.add_argument("--risk_penalty_lambda", type=float, default=0.0, help="Penalty lambda for risk constraint violations")
    p.add_argument("--max_drawdown", type=float, default=float("inf"), help="Maximum allowable drawdown over an episode. inf disables.")

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