from dataclasses import dataclass
import pickle
from typing import Any, Callable

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DemandPerturbationConfig:
    scales: list[float]
    seeds: int
    episodes: int
    out_csv: str = "demand_perturbation.csv"
    out_png: str = "demand_perturbation.png"


def _state_to_key(obs) -> tuple[int, ...]:
    # obs is a pandas Series in your pipeline
    if hasattr(obs, "to_numpy"):
        vals = obs.to_numpy()
    else:
        vals = np.asarray(obs)
    return tuple(int(v) for v in vals)


def evaluate_saved_policy(
    *,
    build_world_fn: Callable[[], tuple[Any, Any, Any, pd.DataFrame, pd.DataFrame]],
    inject_forecast_fn: Callable,
    args: Any,
    policy_path: str,
    q_table_path: str | None,
    n_episodes: int,
    seed: int,
) -> float:
    """
    Returns mean cumulative reward across episodes for this seed.
    """
    np.random.seed(seed)

    state, action_space, market_model, load_df, lmp_df = build_world_fn()

    # load deterministic policy (dict: state_key -> action_idx)
    with open(policy_path, "rb") as f:
        policy_map = pickle.load(f)

    q_table = None
    if q_table_path:
        try:
            with open(q_table_path, "rb") as f:
                q_table = pickle.load(f)
        except FileNotFoundError:
            q_table = None

    # episode start indices (same style as your train loop)
    episode_starts = list(range(0, len(state.raw_state_data) - state.window_size + 1, state.step_hours))
    episode_starts = episode_starts[:n_episodes]

    price_q = float(lmp_df["lmp"].abs().quantile(args.max_notional_q))
    max_notional = float(50 * price_q)  # MAX_BID_QUANTITY_MW in your file is 50

    ep_rewards = []

    for start_idx in episode_starts:
        # inject forecast for this episode (your existing flow)
        forecast_df = ...  # whatever you already compute in train() for forecast
        inject_forecast_fn(state, forecast_df)

        state.apply()
        obs = state.reset(new_episode=False)
        state.ptr = start_idx  # if your State supports direct pointer control; otherwise ignore this line

        done = False
        cumulative_reward = 0.0

        # start
        obs = state.reset(new_episode=False)
        state_key = _state_to_key(obs)

        step_counter = 0
        while not done:
            # choose action
            if state_key in policy_map:
                action_idx_raw = int(policy_map[state_key])
            elif q_table is not None and state_key in q_table:
                action_idx_raw = int(np.argmax(q_table[state_key]))
            else:
                action_idx_raw = 0  # safe fallback

            action_idx, clip_info = action_space.project_to_feasible(
                action_idx_raw,
                max_quantity=50,
                max_notional=max_notional,
            )

            ts = state.timestamps[state.ptr]
            delivery_time = ts + pd.Timedelta(hours=24)

            if delivery_time not in state.raw_state_data.index:
                done = True
                break

            delivery_row = state.raw_state_data.loc[delivery_time]
            price_val = float(delivery_row.get("lmp", np.nan))
            if not np.isfinite(price_val):
                price_val = float(lmp_df["lmp"].iloc[-1])

            _, _, reward = market_model.clear_market_from_action(action_idx, price_val)

            # NOTE: evaluation-only; you probably want no risk penalty shaping here.
            cumulative_reward += float(reward)

            next_obs, _, done, _ = state.step()
            state_key = _state_to_key(next_obs)
            step_counter += 1
            if step_counter >= state.window_size:
                done = True

        ep_rewards.append(cumulative_reward)

    return float(np.mean(ep_rewards)) if ep_rewards else float("nan")

def run_demand_perturbation_sweep(
    *,
    build_world_and_data_fn: Callable[[], tuple[Any, Any, Any, pd.DataFrame, pd.DataFrame]],
    inject_forecast_fn: Callable,
    args: Any,
    cfg: DemandPerturbationConfig,
    policy_path: str,
    q_table_path: str | None,
) -> None:
    rows = []

    for scale in cfg.scales:
        for s in range(cfg.seeds):
            # set scale for this eval run
            args.demand_scale = float(scale)

            mean_reward = evaluate_saved_policy(
                build_world_fn=build_world_and_data_fn,
                inject_forecast_fn=inject_forecast_fn,
                args=args,
                policy_path=policy_path,
                q_table_path=q_table_path,
                n_episodes=cfg.episodes,
                seed=s,
            )
            rows.append({"scale": float(scale), "seed": int(s), "mean_episode_reward": float(mean_reward)})

    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_csv, index=False)

    summary = df.groupby("scale")["mean_episode_reward"].agg(["mean", "std"]).reset_index()

    plt.figure()
    x = summary["scale"].to_numpy()
    m = summary["mean"].to_numpy()
    sd = summary["std"].to_numpy()
    plt.plot(x, m, marker="o")
    plt.fill_between(x, m - sd, m + sd, alpha=0.2)
    plt.xlabel("Demand scale multiplier")
    plt.ylabel("Mean episode reward")
    plt.title("Demand curve perturbations: performance vs demand scale")
    plt.tight_layout()
    plt.savefig(cfg.out_png, dpi=200)
    plt.close()

    print("\n=== Demand perturbation summary ===")
    print(summary)
    print(f"\nSaved CSV: {cfg.out_csv}")
    print(f"Saved plot: {cfg.out_png}")