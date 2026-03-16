from dataclasses import dataclass
from typing import Any, Callable, Optional

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ElasticityPerturbationConfig:
    seeds: int = 5
    episodes: int = 50
    elasticities: Optional[list[float]] = None
    out_csv: str = "elasticity_perturbation.csv"
    out_png: str = "elasticity_perturbation.png"


def run_elasticity_perturbation(
    *,
    train_fn: Callable[..., tuple[Any, Any, Any, Any, list[dict]]],
    args: Any,
    cfg: ElasticityPerturbationConfig,
) -> None:
    if cfg.elasticities is None:
        cfg.elasticities = [0.0, 0.05, 0.1, 0.15, 0.2]

    rows: list[dict] = []
    total_runs = len(cfg.elasticities) * cfg.seeds
    run_num = 0

    for elasticity in sorted(cfg.elasticities):
        for seed in range(cfg.seeds):
            run_num += 1
            print(f"\n========== Elasticity Perturbation Run {run_num}/{total_runs} ==========")
            print(f"elasticity={elasticity} | seed={seed} | episodes={cfg.episodes}")

            overrides = {
                "lmp_competition_elasticity": float(elasticity),
            }
            _, _, _, _, logs = train_fn(
                n_episodes=cfg.episodes,
                seed=seed,
                overrides=overrides,
            )
            for r in logs:
                r["seed"] = int(seed)
                r["elasticity"] = float(elasticity)
            rows.extend(logs)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_csv, index=False)

    summary = (
        df.groupby(["elasticity", "episode"])["cumulative_reward"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    for elasticity in sorted(df["elasticity"].unique()):
        sub = summary[summary["elasticity"] == elasticity].sort_values("episode")
        x = sub["episode"].to_numpy()
        m = sub["mean"].to_numpy()
        sd = sub["std"].to_numpy()
        plt.plot(x, m, label=f"elasticity={elasticity}")
        plt.fill_between(x, m - sd, m + sd, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title("Elasticity perturbation: cumulative reward vs episode")
    plt.legend(title="elasticity", loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(cfg.out_png, dpi=200)
    plt.close()

    final = (
        df.sort_values("episode")
        .groupby(["elasticity", "seed"])
        .tail(1)
    )
    print("\n=== Elasticity Perturbation Summary (final episode reward) ===")
    print(final.groupby("elasticity")["cumulative_reward"].agg(["mean", "std"]))
    print(f"\nSaved CSV: {cfg.out_csv}")
    print(f"Saved plot: {cfg.out_png}")
