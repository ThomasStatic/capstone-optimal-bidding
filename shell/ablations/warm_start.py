from dataclasses import dataclass
from typing import Any, Callable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def _episodes_to_reach_fraction_of_final(rewards: np.ndarray, frac: float = 0.9, tail_k: int = 5) -> int:
    """Convergence speed metric: number of episodes to reach a fraction of the final reward"""
    if rewards.size == 0:
        return -1
    k = int(min(tail_k, rewards.size))
    final = float(np.mean(rewards[-k:]))
    target = frac * final
    for i, r in enumerate(rewards):
        if r >= target:
            return int(i)
    return int(rewards.size - 1)

@dataclass
class WarmStartAblationConfig:
    seeds: int = 5
    episodes: int = 50
    out_csv: str = "warm_start_ablation.csv"
    out_png: str = "warm_start_ablation.png"
    frac_final: float = 0.9
    tail_k: int = 5

def run_warm_start_ablation(
    *,
    train_fn: Callable[..., tuple[Any, Any, Any, Any, list[dict]]],
    args: Any,
    cfg: WarmStartAblationConfig,
) -> None:
    all_rows: list[dict] = []

    for warm in [False, True]:
        for s in range(cfg.seeds):
            args.warm_start_q = warm
            _, _, _, _, logs = train_fn(n_episodes=cfg.episodes, seed=s)
            all_rows.extend(logs)

    df = pd.DataFrame(all_rows)
    df.to_csv(cfg.out_csv, index=False)

    # Aggregate learning curves across seeds
    g = df.groupby(["warm_start_q", "episode"])["cumulative_reward"]
    summary = g.agg(["mean", "std"]).reset_index()

    # Convergence + final reward summaries per seed
    conv_rows = []
    for warm in [False, True]:
        for s in range(cfg.seeds):
            r = (
                df[(df["warm_start_q"] == warm) & (df["seed"] == s)]
                .sort_values("episode")["cumulative_reward"]
                .to_numpy()
            )
            conv_rows.append({
                "warm_start_q": warm,
                "seed": s,
                "final_reward": float(r[-1]) if r.size else np.nan,
                "episodes_to_frac_final": _episodes_to_reach_fraction_of_final(
                    r, frac=cfg.frac_final, tail_k=cfg.tail_k
                ),
            })
    conv = pd.DataFrame(conv_rows)
    
    plt.figure()
    for warm in [False, True]:
        sub = summary[summary["warm_start_q"] == warm].sort_values("episode")
        x = sub["episode"].to_numpy()
        m = sub["mean"].to_numpy()
        sd = sub["std"].to_numpy()
        plt.plot(x, m, label=f"warm_start_q={warm}")
        plt.fill_between(x, m - sd, m + sd, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title("Warm-start ablation: learning curves (mean ± std across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_png, dpi=200)
    plt.close()

    print("\n=== Warm-start ablation summary ===")
    print(conv.groupby("warm_start_q")[["final_reward", "episodes_to_frac_final"]].agg(["mean", "std"]))
    print(f"\nSaved CSV: {cfg.out_csv}")
    print(f"Saved plot: {cfg.out_png}")