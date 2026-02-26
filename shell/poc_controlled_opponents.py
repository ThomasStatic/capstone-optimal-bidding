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

    n_agents: int = 1

    opponent_markup: float = 0.10

    opponent_quantile: float = 0.7 

    temperature_mode: str = "fixed"
    temperature: float = 1.0
    temperature_min: float = 0.1
    temperature_decay: float = 0.995

    policy_freeze_enabled: bool = True
    policy_freeze_k: int = 10
    policy_inertia_keep: float = 0.9

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

## Wrap class for baseline agent
class BaselineAgent:
    def __init__(self, policy, name: str):
        self.policy = policy
        self.name = name
        self.Q = {}
    def act(self, obs):                        return int(self.policy.act(obs))
    def state_to_key(self, obs):               return obs
    def update_q_table(self, *args, **kwargs):  pass
    def extract_policy(self, **kwargs):         return {}

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

    # RL agent (multiple)
    rl_agents = [TabularQLearningAgent(num_actions=action_space.n_actions) for _ in range(cfg.n_agents)]
    for i, ag in enumerate(rl_agents):
        if hasattr(ag, "seed"):
            ag.seed(cfg.seed + i)

    # Replace opponent_policy and opponent_policy_quantile with:
    marginal_cost = float(market_model.params.marginal_cost)
    baselines = [
        BaselineAgent(CostPlusMarkupPolicy(action_space=action_space, marginal_cost=marginal_cost,
                    markup=float(marginal_cost * (1.0 + cfg.opponent_markup)), quantity_mw=MAX_BID_QUANTITY_MW), "cost_plus"),
        BaselineAgent(HistoricalQuantilePolicy(action_space=action_space, lmp_df=lmp_df,
                    quantile=float(cfg.opponent_quantile), quantity_mw=MAX_BID_QUANTITY_MW), "hist_quantile"),
    ]
    all_agents = rl_agents + baselines
    agent_names = [f"RL_Agent_{i}" for i in range(len(rl_agents))] + ["Cost Plus Markup Baseline Agent", "Quantile Baseline Agent"]

    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been initialized")

    all_episode_starts = list(range(0, len(state.raw_state_data) - state.window_size + 1, state.step_hours))

    episode_starts = all_episode_starts[:cfg.n_episodes]

    rewards_ep_rl: List[float] = []
    # baselines
    rewards_ep_costplus: List[float] = []
    rewards_ep_quantile: List[float] = []

    # metrics
    metrics = MetricsTracker(n_agents=len(all_agents), agent_names=agent_names, action_space=action_space)

    mean_td_ep: List[float] = []
    delta_q_ep: List[float] = []
    entropy_ep: List[float] = []

    # logs: List[Dict[str, Any]] = []

    rng = np.random.default_rng(cfg.seed)
    agent_policies: List[dict] = [{} for _ in rl_agents]

    for ep_idx, start_idx in enumerate(episode_starts):
        if cfg.policy_freeze_enabled and cfg.policy_freeze_k > 0 and (ep_idx % cfg.policy_freeze_k) == 0:
            for i, ag in enumerate(rl_agents):
                new_pol = ag.extract_policy(temperature=1e-6)
                old_pol = agent_policies[i]
                merged = {}
                for s in set(old_pol) | set(new_pol):
                    old_a, new_a = old_pol.get(s), new_pol.get(s)
                    if old_a is None:               merged[s] = new_a
                    elif new_a is None:             merged[s] = old_a
                    elif old_a == new_a:            merged[s] = new_a
                    elif rng.random() < cfg.policy_inertia_keep: merged[s] = old_a
                    else:                           merged[s] = new_a
                agent_policies[i] = merged
                if cfg.verbose:
                    frozen_states = len(merged)
                    q_states = len(ag.Q)
                    coverage = frozen_states / max(q_states, 1) * 100
                    print(f"  [Freeze ep={ep_idx}] Agent {i}: frozen_states={frozen_states} | q_states={q_states} | coverage={coverage:.1f}%")
        state.episode_start = start_idx
        episode_start_ts = state.raw_state_data.index[start_idx]

        h_end = episode_start_ts + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
        f_slice = forecast_series.loc[(forecast_series.index >= episode_start_ts) & (forecast_series.index <= h_end)]
        forecast_df = f_slice.rename(FORECAST_LOAD_COL).reset_index()
        inject_episode_forecast(state, forecast_df, demand_scale=float(cfg.demand_scale))

        state.apply()
        obs = state.reset(new_episode=False)
        state_key = rl_agents[0].state_to_key(obs)

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

            action_indices, clip_infos = [], []
            for j, ag in enumerate(all_agents):
                if cfg.policy_freeze_enabled and j < len(rl_agents):
                    # Use frozen policy if state is known, else fall back to softmax
                    frozen_action = agent_policies[j].get(state_key)
                    raw = frozen_action if frozen_action is not None else ag.act(step_obs)
                else:
                    raw = ag.act(step_obs)
                aidx, clip_info = action_space.project_to_feasible(
                    raw, max_quantity=MAX_BID_QUANTITY_MW, max_notional=max_notional
                )
                action_indices.append(int(aidx))
                clip_infos.append(clip_info)

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
            r_rl = float(np.max(rewards[:cfg.n_agents]))
            r_costplus = float(rewards[cfg.n_agents])
            r_quantile = float(rewards[cfg.n_agents + 1])

            q_snapshots = {i: np.array(ag.Q[state_key], dtype=float)
               for i, ag in enumerate(rl_agents) if state_key in ag.Q}

            next_obs, _, done, info = state.step()
            next_state_key = rl_agents[0].state_to_key(next_obs)

            for i, ag in enumerate(rl_agents):
                ag.update_q_table(state_key, action_indices[i], rewards[i], next_state_key, done)
                if i in q_snapshots and state_key in ag.Q:
                    dq_accum += float(np.mean(np.abs(np.array(ag.Q[state_key], dtype=float) - q_snapshots[i])))
                    greedy_action = int(np.argmax(ag.Q[state_key]))
                if state_key in ag.Q:
                    ent_accum += softmax_entropy(np.array(ag.Q[state_key], dtype=float), temperature=temp)
                    ent_count += 1
                     # Greedy episode logging
                    greedy_action = int(np.argmax(ag.Q[state_key]))
                else:
                    greedy_action = 0

                metrics.log_greedy_actions(i, greedy_action)
                if hasattr(ag, "last_td_error"):
                    try: td_errors.append(float(ag.last_td_error))
                    except Exception: pass

            # Metrics logging
            metrics.log_step(
                episode=ep_idx, step=step_counter, timestamp=ts,
                action_indices=action_indices,
                rewards=rewards,
                clearing_price=clearing_price, demand_mw=demand_mw,
                rho=out["rho"],
                clip_infos=clip_infos,
            )

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

        # logs.append(
        #     {
        #         "episode": ep_idx,
        #         "start_idx": int(start_idx),
        #         "max_reward": float(cumulative_reward_rl),
        #         "mean_td_error": float(mean_td_ep[-1]),
        #         "mean_delta_q": float(delta_q_ep[-1]),
        #         "mean_entropy": float(entropy_ep[-1]),
        #         "steps": int(step_counter),
        #     }
        # )

        # # At the end of each episode - close episode and log greedy actions
        # for j, ag in enumerate(rl_agents):
        #     if ag.Q:
        #         greedy_per_state = [
        #             int(np.argmax(q_vals))
        #             for q_vals in ag.Q.values()
        #             if np.any(q_vals != 0)   # only states with real updates
        #         ]
        #         for g in greedy_per_state:
        #             metrics.log_greedy_actions(j, g)

        if cfg.verbose:
        # Diagnostics
            if (ep_idx + 1) % 10 == 0:
                for j, ag in enumerate(rl_agents):
                    n_seen = sum(1 for q in ag.Q.values() if np.any(q != 0))
                    greedy = [int(np.argmax(q)) for q in ag.Q.values() if np.any(q != 0)]
                    unique = len(set(greedy))
                    print(f"  Agent {j}: Q-states={len(ag.Q)} | seen={n_seen} | unique_greedy={unique}")
        metrics.close_episode(ep_idx)   

        if (ep_idx + 1) % 10 == 0:
            print(f"[PoC] ep {ep_idx+1}/{len(episode_starts)} | max_reward_rl={cumulative_reward_rl:.3f} | reward_costplus={cumulative_reward_costplus:.3f} | reward_quantile={cumulative_reward_quantile:.3f}")

    # Create total rewards_epi
    rewards_ep = {"RL Agent": rewards_ep_rl, "Cost Plus": rewards_ep_costplus, "Quantile": rewards_ep_quantile}

    # generate metric level-plots
    export_multi_agent_metrics(metrics, out_dir=cfg.out_dir)

    with open(os.path.join(cfg.out_dir, "poc_q_table.pkl"), "wb") as f:
        pickle.dump([ag.Q for ag in rl_agents], f)

    # 4 plots
    # save_plot_rewards(rewards_ep, "Episode Rewards across agents", os.path.join(cfg.out_dir, "01_reward.png"))
    # save_plot(mean_td_ep, "Mean TD Error (episode)", os.path.join(cfg.out_dir, "02_mean_td_error.png"))
    # save_plot(delta_q_ep, "Mean |ΔQ| per step (episode)", os.path.join(cfg.out_dir, "03_delta_q.png"))
    # save_plot(entropy_ep, "Policy Entropy (episode)", os.path.join(cfg.out_dir, "04_entropy.png"))

    print(f"PoC complete. Outputs saved to: {cfg.out_dir}")
    # print("   - poc_logs.csv")
    # print("   - 01_reward.png, 02_mean_td_error.png, 03_delta_q.png, 04_entropy.png")
    # print("   - poc_q_table.pkl")


if __name__ == "__main__":
    cfg = PoCConfig(
        n_episodes=200,        
        seed=3,
        opponent_markup=0.10,  
        demand_scale=1.0,
        verbose=True,
        n_agents=3
    )
    run_poc(cfg)