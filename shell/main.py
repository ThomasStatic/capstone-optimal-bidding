import argparse
import pandas as pd
import numpy as np
import pickle

from shell.ablations.demand_perturbation import DemandPerturbationConfig, run_demand_perturbation_sweep
from shell.ablations.run_all import AllAblationsConfig, run_all_ablations
from shell.ablations.warm_start import WarmStartAblationConfig, run_warm_start_ablation
from shell.api_controllers.market_loads_api import ISODemandController
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
from typing import TypeAlias

ISO = "ERCOT"
START_DATE = "2025-01-01"
END_DATE = "2026-01-31"

NUM_DISCRETIZER_BINS = 8
MAX_BID_QUANTITY_MW = 50

FORECAST_CSV_PATH = "ERCOT - Load Forecast 2025.csv"
FORECAST_HORIZON_HOURS = 24 * 14  # 2 weeks

StateKey: TypeAlias = tuple[int, ...]

def _load_ercot_forecast_series() -> pd.Series:
    """
    Returns an hourly forecast series indexed by UTC datetime, in MW.
    Uses TOTAL_SYSTEM_LOAD columns and sums Houston+North+South+West.
    """
    df = pd.read_csv(
        FORECAST_CSV_PATH,
        na_values=["-", " -", " -   ", " -   -", "—"],
    )

    df = df.rename(columns={"Date_Time_UTC": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Pick the 4 regional total system load columns
    cols = [
        "ERCOT_HOUSTON_TOTAL_SYSTEM_LOAD (MWh)",
        "ERCOT_NORTH_TOTAL_SYSTEM_LOAD (MWh)",
        "ERCOT_SOUTH_TOTAL_SYSTEM_LOAD (MWh)",
        "ERCOT_WEST_TOTAL_SYSTEM_LOAD (MWh)",
    ]

    # Clean numeric (handles "13,896.77" style strings)
    for c in cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

    df["forecast_total_mw"] = df[cols].sum(axis=1)

    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    end_ts = pd.Timestamp(END_DATE, tz="UTC")

    s = (
        df.loc[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts), ["datetime", "forecast_total_mw"]]
          .dropna()
          .set_index("datetime")["forecast_total_mw"]
          .sort_index()
    )

    return s

def apply_demand_scale_to_state(state: State, scale: float) -> None:
    if scale == 1.0:
        return
    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data not initialized")

    df = state.raw_state_data

    if HIST_LOAD_COL in df.columns:
        df[HIST_LOAD_COL] = df[HIST_LOAD_COL].astype(float) * scale

    if FORECAST_LOAD_COL in df.columns:
        df[FORECAST_LOAD_COL] = df[FORECAST_LOAD_COL].astype(float) * scale


def get_episode_temperature(ep_idx: int) -> float | None:
    """
    Returns:
      - float for fixed/exp_decay
      - None for qgap (agent computes adaptive temperature internally)
    """
    if args.temperature_mode == "fixed":
        return float(args.temperature)

    if args.temperature_mode == "qgap":
        return None

    if args.temperature_mode == "exp_decay":
        T0 = float(args.temperature)
        decay = float(args.temperature_decay)
        Tmin = float(args.temperature_min)
        return max(Tmin, T0 * (decay ** ep_idx))

    raise ValueError(f"Unknown temperature_mode: {args.temperature_mode}")

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

    if args.verbose:
        print("LMP min/max:", price_df["datetime"].min(), price_df["datetime"].max())
        print("Load columns:", list(load_df.columns))
        ts_col = "datetime" if "datetime" in load_df.columns else "period"
        print("Load min/max:", load_df[ts_col].min(), load_df[ts_col].max())
        print("LMP tz:", price_df["datetime"].dtype)
        print("Load tz:", load_df["period"].dtype)

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

    if float(args.demand_scale) != 1.0 and FORECAST_LOAD_COL in forecast_df.columns:
        forecast_df[FORECAST_LOAD_COL] = forecast_df[FORECAST_LOAD_COL].astype(float) * float(args.demand_scale)

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
    lmp_df = pd.read_csv(LMP_CSV_PATH, parse_dates=["CloseDateUTC"])
    lmp_df = lmp_df.rename(columns={"CloseDateUTC": "datetime"})
    lmp_df["datetime"] = pd.to_datetime(lmp_df["datetime"], utc=True)
    lmp_df = lmp_df.loc[
        (lmp_df["datetime"] >= pd.to_datetime(START_DATE, utc=True)) &
        (lmp_df["datetime"] <= pd.to_datetime(END_DATE, utc=True))
    ]

    lmp_start = lmp_df["datetime"].min()
    lmp_end   = lmp_df["datetime"].max()
    start = lmp_start.date().isoformat()
    end   = lmp_end.date().isoformat()

    historic_load_api = ISODemandController(start, end, ISO)
    load_df = historic_load_api.get_market_loads()
    load_df =load_df[load_df["respondent"] == ISO].copy()
    load_df["period"] = pd.to_datetime(load_df["period"], utc=True)
    load_df = load_df.sort_values("period")

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

def build_world_and_data():
    load_df, lmp_df = load_data()
    state, action_space, market_model = build_world(load_df, lmp_df)
    
    apply_demand_scale_to_state(state, float(args.demand_scale))
    state.apply()

    return state, action_space, market_model, load_df, lmp_df

from typing import Any, List, Tuple, TypeAlias

def build_agents(n_agents: int, num_actions: int, *, seed: int | None = None) -> List[TabularQLearningAgent]:
    agents: List[TabularQLearningAgent] = []
    for i in range(n_agents):
        a = TabularQLearningAgent(num_actions=num_actions)
        # If your agent has a seed() method, give each agent a different seed for tie-breaking diversity
        if seed is not None and hasattr(a, "seed"):
            a.seed(int(seed) + 1000 * i)
        agents.append(a)
    return agents

def select_and_project_actions(
    agents: List[TabularQLearningAgent],
    state_key: StateKey,
    action_space: ActionSpace,
    *,
    temperature: float | None,
    max_quantity: float,
    max_notional: float,
) -> Tuple[List[int], List[dict]]:
    """
    "Simultaneous" action submission: everyone picks from the same state_key,
    we project each action to feasibility, then return the final action indices.
    """

    action_indices: List[int] = []
    clip_infos: List[dict] = []

    for agent in agents:
        raw_idx = agent.select_softmax_action(state_key, temperature=temperature)
        aidx, clip_info = action_space.project_to_feasible(
            raw_idx,
            max_quantity=max_quantity,
            max_notional=max_notional,
        )
        action_indices.append(int(aidx))
        clip_infos.append(clip_info)

    return action_indices, clip_infos


def train(n_episodes = 20, *, seed: int | None = None, overrides: dict | None = None) -> tuple[List[TabularQLearningAgent], State, ActionSpace, MarketModel, list[dict]]:
    # For ablation (temporary change for single run)
    if overrides is not None:
        for k, v in overrides.items():
            setattr(args, k, v)

    load_df, lmp_df = load_data()
    state, action_space, market_model = build_world(load_df, lmp_df)
    apply_demand_scale_to_state(state, float(args.demand_scale))
    state.apply()

    if seed is not None:
        np.random.seed(seed)

    price_q = float(lmp_df[PRICE_COL].abs().quantile(args.max_notional_q))
    max_notional = float(MAX_BID_QUANTITY_MW * price_q)

    n_agents = int(getattr(args, "n_agents", 1))
    agents = build_agents(n_agents, num_actions=action_space.n_actions, seed=seed)

    if(args.verbose):
        print(f"[Risk] max_notional_q={args.max_notional_q} | price_q={price_q:.4f} | max_notional={max_notional:.4f}")

    ##agent = TabularQLearningAgent(num_actions=action_space.n_actions)

    ##if seed is not None and hasattr(agent, "seed"):
      ##  agent.seed(seed)

    marginal_cost = float(market_model.params.marginal_cost)
    cost_plus_policy = CostPlusMarkupPolicy(
        action_space=action_space,
        marginal_cost=marginal_cost,
        markup=float(args.markup),
        quantity_mw=MAX_BID_QUANTITY_MW,
    )
    baseline_action_idx = int(cost_plus_policy.act(None))

    episode_logs: list[dict] = []

    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been intialized")
    
    episode_starts = list(range(0, len(state.raw_state_data) - state.window_size + 1, state.step_hours))
    episode_starts = episode_starts[:n_episodes] # Limit to n_episodes

    for ep_idx, start_idx in enumerate(episode_starts):
        if args.verbose:
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
        #sarimax = SARIMAXLoadProjections(history_for_model)
        #forecast_df = sarimax.get_forecast_df()

        #inject_epsisode_forecast(state, forecast_df)

        # Re-discretize now that we have forecast data
        state.apply()
        obs = state.reset(new_episode=False)
        state_key = agents[0].state_to_key(obs)

        done = False
        step_counter = 0

        while not done:
            temp = get_episode_temperature(ep_idx)
            ##action_idx_raw = agent.select_softmax_action(state_key, temperature=temp)

            action_indices, clip_infos = select_and_project_actions(
                agents,
                state_key,
                action_space,
                temperature=temp,
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
        
            # Demand for residual clearing: use realized load at delivery_time if available
            demand_mw = delivery_row.get(HIST_LOAD_COL, np.nan)
            demand_mw = float(demand_mw) if pd.notna(demand_mw) else 0.0

            if args.warm_start_q:
                for ag in agents:
                    if state_key not in ag.Q:
                        baseline_reward = market_model.peek_reward_from_action(baseline_action_idx, float(price_val))

                        margin = 1.0
                        ag.warm_start_state(
                            state_key,
                            preferred_action=baseline_action_idx,
                            preferred_q=baseline_reward + margin,
                            other_q=baseline_reward - margin,
                            only_if_unseen=True,
                        )

            clearing_price = market_model.sample_clearing_price(float(price_val))
            out = market_model.clear_market_multi_agent_residual(
                action_indices,
                clearing_price=clearing_price,
                demand_mw=demand_mw,
                rho_min = float(args.rho_min),
                rho_max = float(args.rho_max),
                rho_k= float(args.rho_k),
                rho_p0= float(args.rho_p0),
                tie_break_random=True,
            )
            rewards = list(out["rewards"])

            if args.risk_penalty_lambda > 0.0:
                for i, clip_info in enumerate(clip_infos):
                    if bool(clip_info.get("clipped", False)):                        
                        orig_notional = float(clip_info["original_notional"])
                        if max_notional > 1e-12:
                            severity = max(0.0, (orig_notional - max_notional) / max_notional)
                            if args.verbose: 
                                print(f"[Risk Penalty] step={step_counter} | orig_notional={orig_notional:.4f} | max_notional={max_notional:.4f} | severity={severity:.4f}")
                                
                        else:
                            severity = 0.0
                        penalty = float(args.risk_penalty_lambda * (1.0 + severity))
                        rewards[i] -= penalty

            next_obs, _, done, info = state.step()
            next_state_key = agents[0].state_to_key(next_obs)

            for i, ag in enumerate(agents):
                ag.update_q_table(
                    state_key,
                    int(action_indices[i]),
                    float(rewards[i]),
                    next_state_key,
                    done
                )

            cumulative_reward += float(np.sum(rewards))

            if args.verbose:
                print(
                    f"[EP {ep_idx+1} | Step {step_counter}] ts={ts} | "
                    f"state_key={state_key} | "
                    f"P={out['clearing_price']:.2f} | demand={out['demand_mw']:.2f} | rho={out['rho']:.3f} | "
                    f"actions={action_indices} | rewards={np.round(rewards, 4).tolist()} | "
                    f"cum_total_reward={cumulative_reward:.4f}"
                )

            state_key = next_state_key
            obs = next_obs
            step_counter += 1

            if step_counter >= state.window_size:
                done = True
            
        if args.verbose:
            print(f"Episode {ep_idx+1} finished after {step_counter} steps")
        
        # lightweight progress (works even when verbose is off)
        if args.progress_every and ((ep_idx + 1) % args.progress_every == 0):
            seed_str = seed if seed is not None else "NA"
            print(f"\n=== SEED {seed_str} | EPISODE {ep_idx+1}/{len(episode_starts)} | start_idx={start_idx} ===")
        
        episode_logs.append({
            "seed": seed if seed is not None else -1,
            "warm_start_q": bool(args.warm_start_q),
            "episode": ep_idx,
            "cumulative_reward": float(cumulative_reward),
        })
         
    with open("q_table.pkl", "wb") as f:
        pickle.dump([ag.Q for ag in agents], f)

    with open("policy.pkl", "wb") as f:
        pickle.dump([ag.extract_softmax_policy() for ag in agents], f)
    
    print("Training complete. Q-table and policy saved.")
    return agents, state, action_space, market_model, episode_logs

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

    p.add_argument(
        "--warm_start_q",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm start unseen-state Q values using the cost+markup baseline (default: enabled).",
    )
    
    p.add_argument(
        "--run_all_ablations",
        action="store_true",
        help="Run all ablation studies (warm start, risk constraint, temperature) and save CSV/plots."
    )
    p.add_argument("--ablation_seeds", type=int, default=5)
    p.add_argument("--ablation_episodes", type=int, default=50)
    p.add_argument("--ablation_out_csv", type=str, default="warm_start_ablation.csv")
    p.add_argument("--ablation_out_png", type=str, default="warm_start_ablation.png")
    p.add_argument("--temperature_mode", choices=["fixed", "qgap", "exp_decay"], default="fixed")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature_min", type=float, default=0.1)
    p.add_argument("--temperature_decay", type=float, default=0.995)
    p.add_argument("--risk_lambda_on", type=float, default=1.0)

    p.add_argument(
    "--demand_scale",
    type=float,
    default=1.0,
    help="Multiply HIST_LOAD_COL and FORECAST_LOAD_COL by this factor before discretization."
)

    p.add_argument(
        "--plot_demand_perturbation",
        action="store_true",
        help="Evaluate a saved policy under demand scales and save performance-vs-scale plot."
    )

    p.add_argument("--demand_scales", type=str, default="0.9,1.0,1.1",
                help="Comma-separated demand scales for perturbation sweep.")
    p.add_argument("--eval_policy_path", type=str, default="policy.pkl",
                help="Path to saved deterministic policy mapping.")
    p.add_argument("--eval_q_table_path", type=str, default="q_table.pkl",
                help="Optional: Q-table fallback if policy missing a state.")
    
    p.add_argument("--run_master_ablations", action="store_true",
               help="Run warm-start + risk + temperature ablations AND the demand perturbation sweep.")
    
    p.add_argument("--progress_every", type=int, default=25,
               help="Print a progress line every N steps (0 disables).")
    
    p.add_argument("--n_agents", type=int, default=2, help="Number of bidding agents.")

    p.add_argument("--rho_min", type=float, default=0.1, help="rho_min: residual share when price is high")
    p.add_argument("--rho_max", type=float, default=0.9, help="rho_max: residual share when price is low")
    p.add_argument("--rho_k", type=float, default=0.05, help="k: steepness of rho(P)")
    p.add_argument("--rho_p0", type=float, default=50.0, help="p0: switch price of rho(P)")



    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.run_all_ablations or args.run_master_ablations:
        cfg = AllAblationsConfig(
            seeds=args.ablation_seeds,
            episodes=args.ablation_episodes,
            risk_lambda_on=args.risk_penalty_lambda
        )
        run_all_ablations(train_fn=train, args=args, cfg=cfg)

    if args.plot_demand_perturbation or args.run_master_ablations:
        scales = [float(x.strip()) for x in args.demand_scales.split(",")]
        cfg = DemandPerturbationConfig(
            scales=scales,
            seeds=args.ablation_seeds,
            episodes=args.ablation_episodes,
        )
        run_demand_perturbation_sweep(
            build_world_and_data_fn=build_world_and_data,
            inject_forecast_fn=inject_epsisode_forecast,
            args=args,
            cfg=cfg,
            policy_path=args.eval_policy_path,
            q_table_path=args.eval_q_table_path
        )
    
    if args.mode == "train":
        train(n_episodes=args.n_episodes)
    
    if args.mode == "baseline":
        run_baselines(
            baseline=args.baseline,
            n_episodes=args.n_episodes,
            markup=args.markup,
            quantile=args.quantile
        )