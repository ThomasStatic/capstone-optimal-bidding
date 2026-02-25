from __future__ import annotations

import os
import math
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shell.api_controllers.market_loads_api import ISODemandController
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
from shell.agent_interface import Observation
from shell.multi_agent_metrics import MetricsTracker, export_multi_agent_metrics
from shell.state_space import State
from shell.tabular_q_agent import TabularQLearningAgent


ISO = "ERCOT"
START_DATE = "2025-01-01"
END_DATE = "2026-01-31"

NUM_DISCRETIZER_BINS = 8
MAX_BID_QUANTITY_MW = 50

FORECAST_CSV_PATH = "ERCOT - Load Forecast 2025.csv"
FORECAST_HORIZON_HOURS = 24 * 14 


@dataclass
class PoCConfig:
    n_episodes: int = 50  
    seed: int = 0
    demand_scale: float = 1.0
    verbose: bool = False

    opponent_markup: float = 0.10

    opponent_quantile: float = 0.7 

    temperature_mode: str = "fixed"
    temperature: float = 1.0
    temperature_min: float = 0.1
    temperature_decay: float = 0.995

    max_notional_q: float = 0.95
    max_drawdown: float = float("inf") 

    rho_min: float = 0.1
    rho_max: float = 0.9
    rho_k: float = 0.05
    rho_p0: float = 50.0

    out_dir: str = "shell/evaluations/poc_controlled_opponents"


def _load_ercot_forecast_series() -> pd.Series:
    df = pd.read_csv(
        FORECAST_CSV_PATH,
        na_values=["-", " -", " -   ", " -   -", "—"],
        low_memory=False,
    )

    df = df.rename(columns={"Date_Time_UTC": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    cols = [
        "ERCOT_HOUSTON_TOTAL_SYSTEM_LOAD (MWh)",
        "ERCOT_NORTH_TOTAL_SYSTEM_LOAD (MWh)",
        "ERCOT_SOUTH_TOTAL_SYSTEM_LOAD (MWh)",
        "ERCOT_WEST_TOTAL_SYSTEM_LOAD (MWh)",
    ]

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


def build_state_and_discretizers(load_df: pd.DataFrame, price_df: pd.DataFrame, *, verbose: bool) -> tuple[State, dict]:
    load_disc = Discretizer(col=HIST_LOAD_COL, n_bins=NUM_DISCRETIZER_BINS)
    forecast_disc = Discretizer(col=FORECAST_LOAD_COL, n_bins=NUM_DISCRETIZER_BINS)
    price_disc = Discretizer(col=PRICE_COL, n_bins=NUM_DISCRETIZER_BINS)

    state = State(
        plant_id="plant_1",
        discretizers={
            "load": load_disc,
            "forecast": forecast_disc,
            "price": price_disc,
        },
    )

    if verbose:
        print("LMP min/max:", price_df["datetime"].min(), price_df["datetime"].max())
        print("Load columns:", list(load_df.columns))
        ts_col = "datetime" if "datetime" in load_df.columns else "period"
        print("Load min/max:", load_df[ts_col].min(), load_df[ts_col].max())

    state.load_state_data(
        {
            "load": load_df.rename(columns={"period": "datetime"}),
            "price": price_df,
        }
    )

    state.apply()
    return state, {"load": load_disc, "forecast": forecast_disc, "price": price_disc}


def inject_episode_forecast(state: State, forecast_df: pd.DataFrame, *, demand_scale: float) -> None:
    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been initialized")

    df = state.raw_state_data

    if FORECAST_LOAD_COL not in df.columns:
        df[FORECAST_LOAD_COL] = np.nan

    forecast_df = forecast_df.copy()
    forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"], utc=True)

    if demand_scale != 1.0 and FORECAST_LOAD_COL in forecast_df.columns:
        forecast_df[FORECAST_LOAD_COL] = forecast_df[FORECAST_LOAD_COL].astype(float) * float(demand_scale)

    f = forecast_df.set_index("datetime")[FORECAST_LOAD_COL]
    df.loc[f.index, FORECAST_LOAD_COL] = f.values


def make_action_space(lmp_df: pd.DataFrame) -> ActionSpace:
    price_disc = Discretizer(col=PRICE_COL, n_bins=NUM_DISCRETIZER_BINS)
    price_disc.fit(lmp_df[[PRICE_COL]])

    qty_grid = np.linspace(0, MAX_BID_QUANTITY_MW, 100)
    qty_df = pd.DataFrame({"quantity": qty_grid})
    qty_disc = Discretizer(col="quantity", n_bins=NUM_DISCRETIZER_BINS)
    qty_disc.fit(qty_df)

    return ActionSpace(price_disc=price_disc, quantity_disc=qty_disc)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    lmp_df = pd.read_csv(LMP_CSV_PATH, parse_dates=["CloseDateUTC"])
    lmp_df = lmp_df.rename(columns={"CloseDateUTC": "datetime"})
    lmp_df["datetime"] = pd.to_datetime(lmp_df["datetime"], utc=True)
    lmp_df = lmp_df.loc[
        (lmp_df["datetime"] >= pd.to_datetime(START_DATE, utc=True))
        & (lmp_df["datetime"] <= pd.to_datetime(END_DATE, utc=True))
    ]

    lmp_start = lmp_df["datetime"].min()
    lmp_end = lmp_df["datetime"].max()
    start = lmp_start.date().isoformat()
    end = lmp_end.date().isoformat()

    historic_load_api = ISODemandController(start, end, ISO)
    load_df = historic_load_api.get_market_loads()
    load_df = load_df[load_df["respondent"] == ISO].copy()
    load_df["period"] = pd.to_datetime(load_df["period"], utc=True)
    load_df = load_df.sort_values("period")

    return load_df, lmp_df


def infer_market_params_from_lmp(
    lmp_df: pd.DataFrame,
    price_col: str,
    *,
    mc_tail_q: float = 0.25,
    vol_clip_q: float = 0.995,
) -> tuple[float, float]:
    price_series = pd.to_numeric(lmp_df[price_col], errors="coerce").dropna()

    cutoff_price = price_series.quantile(mc_tail_q)
    marginal_cost = price_series[price_series <= cutoff_price].mean()

    diffs = price_series.diff().dropna()
    cap = diffs.abs().quantile(vol_clip_q)
    diffs = diffs.clip(-cap, cap)
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    price_noise_std = mad * 1.4826

    if not np.isfinite(marginal_cost):
        marginal_cost = float(price_series.quantile(0.10))
    if not np.isfinite(price_noise_std):
        price_noise_std = float(diffs.std(ddof=0)) if len(diffs) else 0.0

    return float(marginal_cost), float(price_noise_std)


def build_world(load_df: pd.DataFrame, lmp_df: pd.DataFrame, *, verbose: bool) -> tuple[State, ActionSpace, MarketModel]:
    state, _ = build_state_and_discretizers(load_df, lmp_df, verbose=verbose)
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


# Metrics helpers

def softmax_entropy(q: np.ndarray, temperature: float = 1.0) -> float:
    t = max(float(temperature), 1e-6)
    z = q / t
    z = z - np.max(z)
    p = np.exp(z)
    p = p / (np.sum(p) + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def get_episode_temperature(cfg: PoCConfig, ep_idx: int) -> float:
    if cfg.temperature_mode == "fixed":
        return float(cfg.temperature)
    if cfg.temperature_mode == "exp_decay":
        return max(float(cfg.temperature_min), float(cfg.temperature) * (float(cfg.temperature_decay) ** ep_idx))
    return float(cfg.temperature)


def save_plot(y: List[float], title: str, out_path: str) -> None:
    plt.figure()
    plt.plot(y)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_plot_rewards(y: dict[List], title: str, out_path: str) -> None:
    plt.figure()
    for agent_type, rewards in y.items():
        plt.plot(rewards, label=agent_type)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# The PoC training loop:
#  - 1 RL agent learns
#  - 1 opponent is FIXED
# -------------------------
def run_poc(cfg: PoCConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    np.random.seed(cfg.seed)

    load_df, lmp_df = load_data()
    forecast_series = _load_ercot_forecast_series()
    state, action_space, market_model = build_world(load_df, lmp_df, verbose=cfg.verbose)

    apply_demand_scale_to_state(state, float(cfg.demand_scale))
    state.apply()

    price_q = float(lmp_df[PRICE_COL].abs().quantile(cfg.max_notional_q))
    max_notional = float(MAX_BID_QUANTITY_MW * price_q)

    # RL agent (single)
    rl_agent = TabularQLearningAgent(num_actions=action_space.n_actions)
    if hasattr(rl_agent, "seed"):
        rl_agent.seed(cfg.seed)

    # Controlled opponent baseline (Cost plus marketup)
    marginal_cost = float(market_model.params.marginal_cost)
    opponent_policy = CostPlusMarkupPolicy(
        action_space=action_space,
        marginal_cost=marginal_cost,
        markup=float(cfg.opponent_markup),
        quantity_mw=MAX_BID_QUANTITY_MW,
    )

    # Controlled opponent baseline (Quantile)
    opponent_policy_quantile = HistoricalQuantilePolicy(
        action_space = action_space,
        lmp_df= lmp_df,
        quantile = float(cfg.opponent_quantile),
        quantity_mw = MAX_BID_QUANTITY_MW
    )

    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been initialized")

    all_episode_starts = list(range(0, len(state.raw_state_data) - state.window_size + 1, state.step_hours))

    fixed_start = all_episode_starts[0]
    episode_starts = [fixed_start] * cfg.n_episodes

    rewards_ep_rl: List[float] = []
    # baselines
    rewards_ep_costplus: List[float] = []
    rewards_ep_quantile: List[float] = []


    mean_td_ep: List[float] = []
    delta_q_ep: List[float] = []
    entropy_ep: List[float] = []

    logs: List[Dict[str, Any]] = []

    for ep_idx, start_idx in enumerate(episode_starts):
        state.episode_start = start_idx
        episode_start_ts = state.raw_state_data.index[start_idx]

        h_end = episode_start_ts + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
        f_slice = forecast_series.loc[(forecast_series.index >= episode_start_ts) & (forecast_series.index <= h_end)]
        forecast_df = f_slice.rename(FORECAST_LOAD_COL).reset_index()
        inject_episode_forecast(state, forecast_df, demand_scale=float(cfg.demand_scale))

        state.apply()
        obs = state.reset(new_episode=False)
        state_key = rl_agent.state_to_key(obs)

        done = False
        step_counter = 0

        cumulative_reward_rl = 0.0
        td_errors: List[float] = []
        dq_accum = 0.0
        ent_accum = 0.0
        ent_count = 0

        # Baseline
        cumulative_reward_costplus = 0.0
        cumulative_reward_quantile = 0.0

        while not done:
            temp = get_episode_temperature(cfg, ep_idx)
            ts = state.timestamps[state.ptr]

            step_obs: Observation = {
                "timestamp": ts,
                "state_key": state_key,
                "temperature": temp,
            }

            raw_idx = rl_agent.act(step_obs)
            rl_idx, rl_clip = action_space.project_to_feasible(
                raw_idx, max_quantity=MAX_BID_QUANTITY_MW, max_notional=max_notional
            )

            # Baseline - cost plus markup
            opp_raw_idx = int(opponent_policy.act(step_obs))
            opp_idx, _ = action_space.project_to_feasible(
                opp_raw_idx, max_quantity=MAX_BID_QUANTITY_MW, max_notional=max_notional
            )


            # Baseline - quantile
            opp_raw_idx_quant = int(opponent_policy_quantile.act(step_obs))
            opp_idx_quant, _ = action_space.project_to_feasible(
                opp_raw_idx_quant, max_quantity=MAX_BID_QUANTITY_MW, max_notional=max_notional
            )
            action_indices = [int(rl_idx), int(opp_idx), int(opp_idx_quant)]

            delivery_time = ts + pd.Timedelta(hours=24)
            if delivery_time not in state.raw_state_data.index:
                done = True
                break

            delivery_row = state.raw_state_data.loc[delivery_time]
            price_val = delivery_row.get(PRICE_COL, np.nan)
            if pd.isna(price_val):
                price_val = float(lmp_df[PRICE_COL].iloc[-1])

            demand_mw = delivery_row.get(HIST_LOAD_COL, np.nan)
            demand_mw = float(demand_mw) if pd.notna(demand_mw) else 0.0

            clearing_price = market_model.sample_clearing_price(float(price_val))
            out = market_model.clear_market_multi_agent_residual(
                action_indices,
                clearing_price=clearing_price,
                demand_mw=demand_mw,
                rho_min=float(cfg.rho_min),
                rho_max=float(cfg.rho_max),
                rho_k=float(cfg.rho_k),
                rho_p0=float(cfg.rho_p0),
                tie_break_random=True,
            )
            rewards = list(out["rewards"])
            r_rl = float(rewards[0])
            r_costplus = float(rewards[1])
            r_quantile = float(rewards[2])

            q_before = None
            if isinstance(getattr(rl_agent, "Q", None), dict) and state_key in rl_agent.Q:
                q_before = np.array(rl_agent.Q[state_key], dtype=float)

            next_obs, _, done, info = state.step()
            next_state_key = rl_agent.state_to_key(next_obs)
            next_step_obs: Observation = {
                "timestamp": state.timestamps[state.ptr],
                "state_key": next_state_key,
            }

            rl_agent.update(step_obs, int(rl_idx), r_rl, next_step_obs, done)

            if isinstance(info, dict) and "td_error" in info:
                td_errors.append(float(info["td_error"]))
            elif hasattr(rl_agent, "last_td_error"):
                try:
                    td_errors.append(float(getattr(rl_agent, "last_td_error")))
                except Exception:
                    pass

            # ΔQ (state-local)
            if q_before is not None and state_key in rl_agent.Q:
                q_after = np.array(rl_agent.Q[state_key], dtype=float)
                dq_accum += float(np.mean(np.abs(q_after - q_before)))

            # Entropy (state-local)
            if state_key in getattr(rl_agent, "Q", {}):
                ent_accum += softmax_entropy(np.array(rl_agent.Q[state_key], dtype=float), temperature=temp)
                ent_count += 1

            cumulative_reward_rl += r_rl
            state_key = next_state_key
            obs = next_obs
            step_counter += 1

            # Baseline2
            cumulative_reward_costplus += r_costplus
            cumulative_reward_quantile += r_quantile


            if step_counter >= state.window_size:
                done = True

        rewards_ep_rl.append(float(cumulative_reward_rl))
        rewards_ep_costplus.append(float(cumulative_reward_costplus))
        rewards_ep_quantile.append(float(cumulative_reward_quantile))
        mean_td_ep.append(float(np.mean(td_errors)) if len(td_errors) else float("nan"))
        delta_q_ep.append(float(dq_accum / max(step_counter, 1)))
        entropy_ep.append(float(ent_accum / max(ent_count, 1)) if ent_count else float("nan"))

        logs.append(
            {
                "episode": ep_idx,
                "start_idx": int(start_idx),
                "reward": float(cumulative_reward_rl),
                "mean_td_error": float(mean_td_ep[-1]),
                "mean_delta_q": float(delta_q_ep[-1]),
                "mean_entropy": float(entropy_ep[-1]),
                "steps": int(step_counter),
            }
        )

        if (ep_idx + 1) % 10 == 0:
            print(f"[PoC] ep {ep_idx+1}/{len(episode_starts)} | reward_rl={cumulative_reward_rl:.3f} | reward_costplus={cumulative_reward_costplus:.3f} | reward_quantile={cumulative_reward_quantile:.3f}")

    # Save logs + artifacts
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(cfg.out_dir, "poc_logs.csv"), index=False)

    # Create total rewards_epi
    rewards_ep = {"RL Agent": rewards_ep_rl, "Cost Plus": rewards_ep_costplus, "Quantile": rewards_ep_quantile}

    with open(os.path.join(cfg.out_dir, "poc_q_table.pkl"), "wb") as f:
        pickle.dump(rl_agent.Q, f)

    # 4 plots
    save_plot_rewards(rewards_ep, "Episode Rewards across agents", os.path.join(cfg.out_dir, "01_reward.png"))
    save_plot(mean_td_ep, "Mean TD Error (episode)", os.path.join(cfg.out_dir, "02_mean_td_error.png"))
    save_plot(delta_q_ep, "Mean |ΔQ| per step (episode)", os.path.join(cfg.out_dir, "03_delta_q.png"))
    save_plot(entropy_ep, "Policy Entropy (episode)", os.path.join(cfg.out_dir, "04_entropy.png"))

    print(f"PoC complete. Outputs saved to: {cfg.out_dir}")
    print("   - poc_logs.csv")
    print("   - 01_reward.png, 02_mean_td_error.png, 03_delta_q.png, 04_entropy.png")
    print("   - poc_q_table.pkl")


if __name__ == "__main__":
    cfg = PoCConfig(
        n_episodes=300,        
        seed=0,
        opponent_markup=0.10,  
        demand_scale=1.0,
        verbose=False,
    )
    run_poc(cfg)