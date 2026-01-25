from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shell.main import (
    load_data,
    build_world,
    inject_epsisode_forecast,
    MAX_BID_QUANTITY_MW,
)
from shell.load_sarimax_projections import SARIMAXLoadProjections
from shell.linear_approximator import PRICE_COL
from shell.tabular_q_agent import TabularQLearningAgent


def _unpack_step(step_out: Tuple[Any, ...]) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """
    Supports:
      (next_state, reward, done)
      (next_state, reward, done, info)
    Your State.step currently returns 4 values, but this is robust.
    """
    if not isinstance(step_out, tuple):
        raise TypeError(f"env.step(...) must return a tuple, got {type(step_out)}")

    if len(step_out) == 3:
        next_obs, r, done = step_out
        return next_obs, float(r), bool(done), {}
    if len(step_out) == 4:
        next_obs, r, done, info = step_out
        return next_obs, float(r), bool(done), (info if isinstance(info, dict) else {})
    raise ValueError(f"env.step(...) must return 3 or 4 values, got {len(step_out)}")


def _moving_average(x: List[float], window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if window <= 1 or arr.size == 0:
        return arr
    w = min(window, arr.size)
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(arr, kernel, mode="valid")


def _make_run_dir(base_dir: str, run_name: str | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"run_{ts}"
    out = Path(base_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run main.py training with learning metrics + plots (no learning changes).")
    p.add_argument("--n_episodes", type=int, default=30)
    p.add_argument("--moving_avg_window", type=int, default=10)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--outputs_dir", type=str, default="./outputs/metrics")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--show", action="store_true", help="Show matplotlib windows (still saves plots).")
    p.add_argument("--verbose", action="store_true", help="Print step-level info (can be noisy).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_df, lmp_df = load_data()
    state, action_space, market_model = build_world(load_df, lmp_df)

    agent = TabularQLearningAgent(num_actions=action_space.n_actions)

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    exploration_values: List[float] = [] 

    mean_abs_delta_q: List[float] = []
    mean_abs_td_error: List[float] = []

    temperature = 1.0

    out_dir = _make_run_dir(args.outputs_dir, args.run_name)

    if not isinstance(state.raw_state_data, pd.DataFrame):
        raise ValueError("State raw_state_data has not been initialized (build_world should set it).")

    for ep in range(args.n_episodes):
        ts0 = state.timestamps[state.ptr]
        history_for_model = load_df[load_df["period"] < ts0].copy()
        if history_for_model.empty:
            history_for_model = load_df.iloc[: state.window_size].copy()

        sarimax = SARIMAXLoadProjections(history_for_model)
        forecast_df = sarimax.get_forecast_df()

        inject_epsisode_forecast(state, forecast_df)

        state.apply()

        obs = state.reset(new_episode=False)
        state_key = agent.state_to_key(obs)

        done = False
        step_counter = 0
        ep_return = 0.0

        abs_delta_q_sum = 0.0
        abs_td_sum = 0.0
        n_updates = 0

        while not done:
            action_idx_raw = agent.select_softmax_action(state_key)

            action_idx, clip_info = action_space.project_to_feasible(
                action_idx_raw,
                max_quantity=MAX_BID_QUANTITY_MW,
                max_notional=None,  
            )

            ts = state.timestamps[state.ptr]
            delivery_time = ts + pd.Timedelta(hours=24)

            if delivery_time not in state.raw_state_data.index:
                done = True
                break

            row = state.raw_state_data.loc[delivery_time]
            price_val = float(row[PRICE_COL])

            _, _, reward = market_model.clear_market_from_action(action_idx, price_val)

            agent._ensure_state(state_key)  
            q_before = float(agent.Q[state_key][action_idx])

            step_out = state.step(action_idx)
            next_obs, _placeholder_reward, done, _info = _unpack_step(step_out)
            next_state_key = agent.state_to_key(next_obs)

            if done or next_state_key is None:
                target = float(reward)
            else:
                agent._ensure_state(next_state_key)
                target = float(reward + agent.gamma * float(np.max(agent.Q[next_state_key])))

            td_error = target - q_before

            agent.update_q_table(state_key, action_idx, float(reward), next_state_key, bool(done))

            q_after = float(agent.Q[state_key][action_idx])
            delta_q = q_after - q_before

            abs_delta_q_sum += abs(delta_q)
            abs_td_sum += abs(td_error)
            n_updates += 1

            ep_return += float(reward)
            step_counter += 1
            state_key = next_state_key

            if step_counter >= state.window_size:
                done = True

            if args.verbose:
                print(
                    f"[EP {ep+1} | Step {step_counter}] ts={ts} | "
                    f"a(raw->feasible)={action_idx_raw}->{action_idx} | "
                    f"price={price_val:.2f} | r={reward:.4f} | "
                    f"td={td_error:.6f} | dQ={delta_q:.6f}"
                )

        episode_returns.append(ep_return)
        episode_lengths.append(step_counter)
        exploration_values.append(float(temperature))

        if n_updates > 0:
            mean_abs_delta_q.append(abs_delta_q_sum / n_updates)
            mean_abs_td_error.append(abs_td_sum / n_updates)
        else:
            mean_abs_delta_q.append(0.0)
            mean_abs_td_error.append(0.0)

        if (ep + 1) % max(1, args.log_every) == 0:
            print(
                f"Episode {ep+1}/{args.n_episodes} | "
                f"len={episode_lengths[-1]} | return={episode_returns[-1]:.4f} | "
                f"temp={exploration_values[-1]:.2f} | "
                f"mean|ΔQ|={mean_abs_delta_q[-1]:.6f} | "
                f"mean|TD|={mean_abs_td_error[-1]:.6f}"
            )

    # ---- 4 plots ----
    # 1) Episode return vs episode + moving average
    x = np.arange(1, len(episode_returns) + 1)

    plt.figure()
    plt.plot(x, episode_returns, label="episode_return")
    ma = _moving_average(episode_returns, args.moving_avg_window)
    if args.moving_avg_window > 1 and ma.size > 0:
        x_ma = np.arange(args.moving_avg_window, args.moving_avg_window + ma.size)
        plt.plot(x_ma, ma, label=f"moving_avg(w={min(args.moving_avg_window, len(episode_returns))})")
    plt.title("Episode return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "episode_return.png", dpi=150)

    # 2) Exploration (temperature) vs episode
    plt.figure()
    plt.plot(x, exploration_values)
    plt.title("Exploration schedule (temperature)")
    plt.xlabel("Episode")
    plt.ylabel("Temperature")
    plt.tight_layout()
    plt.savefig(out_dir / "exploration_temperature.png", dpi=150)

    # 3) Mean |ΔQ| vs episode
    plt.figure()
    plt.plot(x, mean_abs_delta_q)
    plt.title("Mean |ΔQ| per episode")
    plt.xlabel("Episode")
    plt.ylabel("Mean |ΔQ|")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_abs_delta_q.png", dpi=150)

    # 4) Mean |TD error| vs episode
    plt.figure()
    plt.plot(x, mean_abs_td_error)
    plt.title("Mean |TD error| per episode")
    plt.xlabel("Episode")
    plt.ylabel("Mean |TD error|")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_abs_td_error.png", dpi=150)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print(f"\nSaved 4 plots to: {out_dir}\n"
          f" - episode_return.png\n"
          f" - exploration_temperature.png\n"
          f" - mean_abs_delta_q.png\n"
          f" - mean_abs_td_error.png\n")


if __name__ == "__main__":
    main()
