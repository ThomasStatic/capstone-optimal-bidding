import argparse
import math
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import your train() and parse_args or whatever is needed.
# If train() depends on global "args", import main and set main.args first.
import shell.main as main


def mean_ci_95(x: np.ndarray) -> tuple[float, float]:
    """
    Returns (mean, half_width) for 95% CI using Student-t approx.
    For n>=30, z ~= 1.96. For small n, we use a conservative lookup.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    m = float(x.mean())
    if n == 1:
        return (m, float("nan"))

    s = float(x.std(ddof=1))
    se = s / math.sqrt(n)

    # df-based t critical values (two-sided 95%)
    t_crit_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 24: 2.064, 29: 2.045
    }
    df = n - 1
    if df >= 30:
        t_crit = 1.96
    else:
        closest = min(t_crit_table.keys(), key=lambda k: abs(k - df))
        t_crit = t_crit_table[closest]

    return (m, float(t_crit * se))


def _init_main_args_with_defaults() -> None:
    """
    main.train() expects global main.args, and main.parse_args() reads sys.argv.
    This runner has its own CLI (--seeds/--episodes), so we must prevent those
    flags from being passed into main.parse_args().
    """
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]  # pretend no CLI args were passed to main
        main.args = main.parse_args()
    finally:
        sys.argv = saved_argv


def run_multi_seed(*, seeds: int, episodes: int, out_csv: str, out_png: str, out_steps_csv: str | None = None):
    """
    Fixes:
      - Converts step-logged cumulative_reward into per-episode return by taking the last value
        for each (seed, episode).
      - Produces per-seed learning curves (debug-friendly).
      - Computes final-episode CI across seeds using per-episode return.
      - Optionally saves raw step-level logs if out_steps_csv is provided.
    """
    all_rows: list[dict] = []

    for s in range(seeds):
        _init_main_args_with_defaults()

        # call your training (train currently logs per step)
        _, _, _, _, logs = main.train(n_episodes=episodes, seed=s)

        # ensure seed is correct
        for r in logs:
            r["seed"] = s
        all_rows.extend(logs)

    df_steps = pd.DataFrame(all_rows)

    required = {"seed", "episode", "cumulative_reward"}
    missing = required - set(df_steps.columns)
    if missing:
        raise KeyError(f"Missing required columns in logs: {missing}")

    # Save raw step-level logs (optional)
    if out_steps_csv:
        df_steps.to_csv(out_steps_csv, index=False)
        print(f"Saved step-level CSV: {out_steps_csv}")

    # --- Convert step logs -> per-episode logs ---
    # Each (seed, episode) may have many rows. Episode return = last cumulative_reward in that group.
    # NOTE: This assumes logs are appended in correct time order within each episode (true in your train()).
    df_ep = (
        df_steps.groupby(["seed", "episode"], as_index=False)
        .agg(
            episode_return=("cumulative_reward", "last"),
            n_rows=("cumulative_reward", "size"),
        )
        .sort_values(["seed", "episode"])
        .reset_index(drop=True)
    )

    # Save episode-level CSV (this replaces old out_csv meaning)
    df_ep.to_csv(out_csv, index=False)

    # --- Final episode return per seed ---
    final = df_ep.sort_values("episode").groupby("seed").tail(1)
    finals = final["episode_return"].to_numpy()

    mean_final, ci_hw = mean_ci_95(finals)
    print("\n=== Multi-seed final EPISODE return ===")
    print(f"seeds={seeds}, episodes={episodes}")
    print(f"mean={mean_final:.4f}, 95% CI = [{mean_final-ci_hw:.4f}, {mean_final+ci_hw:.4f}]")
    if finals.size >= 2:
        print(f"std={np.std(finals, ddof=1):.4f}")
    else:
        print("std=nan")

    # Helpful debug: show largest episode return rows
    top = df_ep.sort_values("episode_return", ascending=False).head(5)
    print("\nTop 5 episode returns:")
    print(top[["seed", "episode", "episode_return", "n_rows"]].to_string(index=False))

    # --- Plot per-seed learning curves ---
    plt.figure()
    for s in sorted(df_ep["seed"].unique()):
        sub = df_ep[df_ep["seed"] == s].sort_values("episode")
        if sub.empty:
            continue
        plt.plot(sub["episode"], sub["episode_return"], alpha=0.8, label=f"seed {s}")

    plt.xlabel("Episode")
    plt.ylabel("Episode return (sum of rewards in episode)")
    plt.title("Per-seed learning curves (episode return)")
    if seeds <= 12:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"\nSaved episode-level CSV: {out_csv}")
    print(f"Saved plot: {out_png}")


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out_csv", type=str, default="multi_seed_episode_eval.csv",
                   help="Episode-level CSV (one row per seed+episode)")
    p.add_argument("--out_png", type=str, default="per_seed_learning_curves.png",
                   help="Per-seed learning curve plot")
    p.add_argument("--out_steps_csv", type=str, default="",
                   help="Optional: save raw step-level logs to this CSV (leave blank to skip)")
    args = p.parse_args()

    out_steps_csv = args.out_steps_csv.strip() or None

    run_multi_seed(
        seeds=args.seeds,
        episodes=args.episodes,
        out_csv=args.out_csv,
        out_png=args.out_png,
        out_steps_csv=out_steps_csv,
    )


if __name__ == "__main__":
    main_cli()
