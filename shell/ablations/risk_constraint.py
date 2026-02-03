from dataclasses import dataclass
from typing import Any, Callable

from matplotlib import pyplot as plt
import pandas as pd


@dataclass
class RiskAblationConfig:
    seeds: int = 5
    episodes: int = 50
    lambda_on: float = 1.0
    out_csv: str = "risk_ablation.csv"
    out_png: str = "risk_ablation.png"

def run_risk_constraint_ablation(*, train_fn: Callable, args: Any, cfg: RiskAblationConfig) -> None:
    """Ablation study for risk_penalty_lambda = 0.0 vs risk_penalty_lambda = <some constant>"""
    rows: list[dict] = []

    conditions = [
        {"risk_penalty_lambda": 0.0, "label": "risk_penalty_lambda=0.0"},
        {"risk_penalty_lambda": cfg.lambda_on, "label": f"risk_penalty_lambda={cfg.lambda_on}"},
    ]

    for cond in conditions:
        for s in range(cfg.seeds):
            overrides = {"risk_penalty_lambda": cond["risk_penalty_lambda"]}
            _, _, _, _, logs = train_fn(n_epsiodes=cfg.episodes, seed=s, overrides=overrides)

            # ensure logs contain these columns (you already log warm_start_q etc; add these two lines when creating rows)
            for r in logs:
                r["seed"] = s
                r["risk_penalty_lambda"] = float(cond["risk_penalty_lambda"])
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
    plt.title("Risk constraint ablation: learning curves (mean ± std across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_png, dpi=200)
    plt.close()

    # quick summary
    final = df.sort_values("episode").groupby(["condition", "seed"]).tail(1)
    print("\n=== Risk constraint ablation summary (final episode reward) ===")
    print(final.groupby("condition")["cumulative_reward"].agg(["mean", "std"]))
    print(f"\nSaved CSV: {cfg.out_csv}")
    print(f"Saved plot: {cfg.out_png}")