from dataclasses import dataclass
from typing import Any, Callable

from matplotlib import pyplot as plt
import pandas as pd


@dataclass
class TemperatureAblationConfig:
    seeds: int = 5
    episodes: int = 50
    out_csv: str = "temperature_ablation.csv"
    out_png: str = "temperature_ablation.png"

def run_temperature_ablation(*, train_fn: Callable, args: Any, cfg: TemperatureAblationConfig) -> None:
    rows: list[dict] = []

    conditions = [
        {"label": "fixed_T=1.0", "overrides": {"temperature_mode": "fixed", "temperature": 1.0}},
        {"label": "adaptive_qgap", "overrides": {"temperature_mode": "qgap"}},
        {"label": "exp_decay_T0=1.0", "overrides": {"temperature_mode": "exp_decay", "temperature": 1.0, "temperature_decay": 0.995, "temperature_min": 0.1}},
    ]

    for cond in conditions:
        for s in range(cfg.seeds):
            _, _, _, _, logs = train_fn(n_episodes=cfg.episodes, seed=s, overrides=cond["overrides"])
            for r in logs:
                r["seed"] = s
                r["condition"] = cond["label"]
            rows.extend(logs)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_csv, index=False)

    # learning curves: mean ± std across seeds
    g = df.groupby(["condition", "episode"])["cumulative_reward"]
    summary = g.agg(["mean", "std"]).reset_index()

    plt.figure()
    for cond in summary["condition"].unique():
        sub = summary[summary["condition"] == cond].sort_values("episode")
        x = sub["episode"].to_numpy()
        m = sub["mean"].to_numpy()
        sd = sub["std"].to_numpy()
        plt.plot(x, m, label=cond)
        plt.fill_between(x, m - sd, m + sd, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title("Temperature schedule ablation: learning curves (mean ± std across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_png, dpi=200)
    plt.close()

    final = df.sort_values("episode").groupby(["condition", "seed"]).tail(1)
    print("\n=== Temperature ablation summary (final episode reward) ===")
    print(final.groupby("condition")["cumulative_reward"].agg(["mean", "std"]))
    print(f"\nSaved CSV: {cfg.out_csv}")
    print(f"Saved plot: {cfg.out_png}")