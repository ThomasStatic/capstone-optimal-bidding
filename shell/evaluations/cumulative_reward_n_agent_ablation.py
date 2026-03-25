import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class CumulativeRewardNAgentAblationConfig:
    seeds: int = 5
    episodes: int = 50
    freeze_ks: Optional[list[int]] = None
    n_agents: Optional[list[int]] = None
    out_csv: str = "cumulative_reward_n_agent_ablation.csv"
    out_png: str = "cumulative_reward_n_agent_ablation.png"


def run_cumulative_reward_n_agent_ablation(
    *,
    train_fn: Callable[..., tuple[Any, Any, Any, Any, list[dict]]],
    args: Any,
    cfg: CumulativeRewardNAgentAblationConfig,
) -> None:
    if cfg.freeze_ks is None:
        cfg.freeze_ks = [5]
    if cfg.n_agents is None:
        cfg.n_agents = [1,2,5,10,100,1000]

    rows: list[dict] = []
    total_runs = len(cfg.n_agents) * len(cfg.freeze_ks) * cfg.seeds
    run_num = 0
    started_at = time.time()

    for n_agents in sorted(cfg.n_agents):
        for k in sorted(cfg.freeze_ks):
            for s in range(cfg.seeds):
                run_num += 1
                elapsed = time.time() - started_at
                avg = elapsed / run_num
                remain = avg * (total_runs - run_num)
                remain_str = time.strftime("%H:%M:%S", time.gmtime(remain))

                print(
                    f"\n========== Cumulative Reward for N agents Ablation Run {run_num}/{total_runs} (ETA {remain_str}) =========="
                )
                print(
                    f"agent_run={s+1}/{cfg.seeds} | agents={n_agents} | freeze_k={k} | seed={s} | episodes={cfg.episodes}"
                )

                overrides = {
                    "policy_freeze_k": int(k),
                    "policy_freeze_enabled": True,
                    "n_agents": int(n_agents),
                    "n_episodes": cfg.episodes,
                    "export_metrics": False
                }
                _, _, _, _, logs = train_fn(
                    n_episodes=cfg.episodes,
                    seed=s,
                    overrides=overrides,
                )
                for r in logs:
                    r["seed"] = int(s)
                    r["freeze_k"] = int(k)
                    r["n_agents"] = int(n_agents)
                    r["cumulative_reward_per_agent"] = r["cumulative_reward"] / int(n_agents)
                rows.extend(logs)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_csv, index=False)

    # Aggregate mean/std by freeze_k, n_agents and episode
    summary = (
        df.groupby(["n_agents", "freeze_k", "episode"])["cumulative_reward_per_agent"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(11, 6))
    grouped = summary.groupby(["n_agents", "freeze_k"])
    for (n_agents, k), g in grouped:
        g = g.sort_values("episode")
        x = g["episode"].to_numpy()
        m = g["mean"].to_numpy()
        sd = g["std"].to_numpy()
        plt.plot(x, m, label=f"agents={n_agents} K={k}")
        plt.fill_between(x, m - sd, m + sd, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward per agent")
    plt.title("N Agent ablation: cumulative reward per agent vs episode")
    plt.legend(title="n_agents/K", loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(cfg.out_png, dpi=200)
    plt.close()

    final = (
        df.sort_values("episode")
        .groupby(["n_agents", "freeze_k", "seed"])
        .tail(1)
    )
    print("\n=== Cumulative Reward for N agents Ablation Summary (final episode reward) ===")
    print(final.groupby(["n_agents", "freeze_k"])["cumulative_reward_per_agent"].agg(["mean", "std"]))
    print(f"\nSaved CSV: {cfg.out_csv}")
    print(f"Saved plot: {cfg.out_png}")