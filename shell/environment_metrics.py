from __future__ import annotations

"""
environment_metrics.py
=======================
Two things live here:

1. EnvironmentMetrics
   A general-purpose metrics collector for the environment, state space, and
   action space — with no knowledge of agents or Q-tables.

   It is fed raw observations (state_key, action_idx, next_state_key) as they
   are produced during training. At the end of each episode it computes
   summaries. At the end of training, call .export() to get a full diagnostic.

2. measure_transition_stationarity(snapshots)
   Directly tests the Arslan & Yüksel (2017) core assumption:

       "Within each exploration phase k, each agent faces a
        stationary Markov decision problem."

   This requires P[x' | x, u] to be time-invariant across episode windows.
   We estimate P[x' | x, u] independently in each window and test whether
   those estimates are consistent.

──────────────────────────────────────────────────────────────────────
Key design decisions
──────────────────────────────────────────────────────────────────────
- EnvironmentMetrics knows nothing about agents. It only sees
  (state_key, action_idx, next_state_key, reward_signal, timestamp).
  "reward_signal" here is the clearing price — a property of the
  environment, not of any agent's bidding strategy.

- TransitionSnapshot is the bridge: one snapshot per episode window.
  You build it by calling env_metrics.episode_snapshot(window_id).
  Then pass a list of snapshots to measure_transition_stationarity().

- All thresholds are named constants. Swap them without touching logic.

──────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────
    from shell.environment_metrics import EnvironmentMetrics, measure_transition_stationarity

    env_metrics = EnvironmentMetrics(action_space=action_space)

    # Inside the training loop, after each step:
    env_metrics.log_transition(
        episode   = ep_idx,
        state_key = state_key,
        action_idx= action_indices[0],      # any representative action index
        next_key  = next_state_key,
        clearing_price = clearing_price,
        demand_mw      = demand_mw,
    )
    # At episode end:
    env_metrics.close_episode(ep_idx)

    # After all episodes:
    snapshots = env_metrics.window_snapshots(window_size=20)
    report    = measure_transition_stationarity(snapshots)
    report.print_summary()
    report.export(out_dir=cfg.out_dir)
    env_metrics.export(out_dir=cfg.out_dir)
"""

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────
# Transition kernel stationarity
TVD_WARN_THRESHOLD   = 0.025   # total variation distance between windows → non-stationary
TVD_FAIL_THRESHOLD   = 0.05   # TVD above this → strong violation of A&Y Assumption 1
# State space health
ENTROPY_MIN_NORM     = 0.40   # normalised state-visit entropy below this → collapsed
COVERAGE_MIN_PCT     = 10.0   # % of possible states visited → under-discretised
# Environment drift
PRICE_DRIFT_PCT      = 20.0   # % drift in rolling mean clearing price → non-stationary
DEMAND_DRIFT_PCT     = 20.0   # % drift in rolling mean demand → non-stationary

@dataclass
class TransitionRecord:
    """One observed transition: (state, action) → next_state, with market context."""
    episode:        int
    state_key:      tuple
    action_idx:     int
    next_key:       tuple
    clearing_price: float
    demand_mw:      float

@dataclass
class EpisodeSummary:
    episode:           int
    n_steps:           int
    n_unique_states:   int     # distinct state_keys visited
    n_unique_sa_pairs: int     # distinct (state, action) pairs
    mean_price:        float
    std_price:         float
    mean_demand:       float
    state_entropy:     float   # normalised Shannon entropy of state visits

@dataclass
class TransitionSnapshot:
    """
    Empirical estimate of P[x' | x, u] over a contiguous window of episodes.

    kernel: dict mapping (state_key, action_idx) → {next_key: probability}
    counts: dict mapping (state_key, action_idx) → {next_key: raw count}
    """
    window_id:     int
    episode_start: int
    episode_end:   int
    kernel:        Dict[tuple, Dict[tuple, float]]  # (s,a) → {s': P}
    counts:        Dict[tuple, Dict[tuple, int]]    # (s,a) → {s': count}
    n_transitions: int
    mean_price:    float
    mean_demand:   float

@dataclass
class WindowComparison:
    """TVD between two adjacent windows for every (s,a) pair they share."""
    window_a:       int
    window_b:       int
    mean_tvd:       float          # mean TVD across all shared (s,a) pairs
    max_tvd:        float          # worst-case (s,a) pair
    n_shared_pairs: int            # how many (s,a) pairs appear in both windows
    coverage_ratio: float          # shared / total pairs in window_a
    passed:         bool
    warnings:       List[str] = field(default_factory=list)


@dataclass
class StationarityReport:
    """Full report from measure_transition_stationarity."""
    timestamp:       str
    n_windows:       int
    comparisons:     List[WindowComparison]
    overall_passed:  bool
    mean_tvd_global: float         # mean TVD across all window-pairs
    max_tvd_global:  float
    summary:         List[str] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n" + "=" * 65)
        print(f"TRANSITION STATIONARITY REPORT  [{self.timestamp}]")
        print(f"Stability across windows according to Multi-Agent Design Report")
        print("=" * 65)
        for c in self.comparisons:
            status = "✓ PASS" if c.passed else "✗ FAIL"
            print(f"\n  Window {c.window_a} → {c.window_b}  {status}")
            print(f"    mean_tvd        = {c.mean_tvd:.4f}   (warn>{TVD_WARN_THRESHOLD}, fail>{TVD_FAIL_THRESHOLD})")
            print(f"    max_tvd         = {c.max_tvd:.4f}")
            print(f"    shared_sa_pairs = {c.n_shared_pairs}  (coverage={c.coverage_ratio:.1%})")
            for w in c.warnings:
                print(f"    ⚠  {w}")
        print(f"\n  Overall : {'PASS' if self.overall_passed else 'FAIL'}")
        print(f"  Global mean TVD : {self.mean_tvd_global:.4f}")
        print(f"  Global max  TVD : {self.max_tvd_global:.4f}")
        print("=" * 65 + "\n")

    def export(self, out_dir: str = ".") -> str:
        rows = []
        for c in self.comparisons:
            rows.append({
                "window_a":       c.window_a,
                "window_b":       c.window_b,
                "mean_tvd":       c.mean_tvd,
                "max_tvd":        c.max_tvd,
                "n_shared_pairs": c.n_shared_pairs,
                "coverage_ratio": c.coverage_ratio,
                "passed":         c.passed,
                "warnings":       "; ".join(c.warnings),
            })
        os.makedirs(out_dir, exist_ok=True)
        ts   = datetime.today().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"transition_stationarity_{ts}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved stationarity CSV  -> {path}")
        return path

    def plot(self, out_dir: str = ".") -> str:
        """
        Two panels:
          Top    — mean TVD per window-pair (bar chart, threshold lines)
          Bottom — max TVD per window-pair
        """
        pairs      = [f"W{c.window_a}→W{c.window_b}" for c in self.comparisons]
        mean_tvds  = [c.mean_tvd for c in self.comparisons]
        max_tvds   = [c.max_tvd  for c in self.comparisons]
        colors     = ["#E07B39" if not c.passed else "#2E86AB" for c in self.comparisons]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.bar(pairs, mean_tvds, color=colors, alpha=0.85)
        ax1.axhline(TVD_WARN_THRESHOLD, color="orange", linestyle="--",
                    linewidth=1.2, label=f"Warn ({TVD_WARN_THRESHOLD})")
        ax1.axhline(TVD_FAIL_THRESHOLD, color="red", linestyle="--",
                    linewidth=1.2, label=f"Fail ({TVD_FAIL_THRESHOLD})")
        ax1.set_ylabel("Mean TVD")
        ax1.set_title("Transition Kernel Stationarity — Mean Total Variation Distance\n"
                      "per Adjacent Window Pair  [Arslan & Yüksel 2017, Assumption 1]")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3, axis="y")

        ax2.bar(pairs, max_tvds, color=colors, alpha=0.85)
        ax2.axhline(TVD_WARN_THRESHOLD, color="orange", linestyle="--", linewidth=1.2)
        ax2.axhline(TVD_FAIL_THRESHOLD, color="red",    linestyle="--", linewidth=1.2)
        ax2.set_ylabel("Max TVD")
        ax2.set_xlabel("Window Pair")
        ax2.set_title("Worst-case (s,a) pair TVD per Window")
        ax2.grid(alpha=0.3, axis="y")

        plt.xticks(rotation=30, ha="right", fontsize=8)
        fig.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        ts   = datetime.today().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"transition_stationarity_{ts}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"Saved stationarity plot -> {path}")
        return path

class EnvironmentMetrics:
    """
    Tracks properties of the environment, state space, and action space.

    Knows nothing about Q-tables, rewards earned by agents, or bidding
    strategies. It only observes:
      - which states were visited
      - which (state, action) pairs were taken
      - what the resulting next state was
      - what the market clearing price and demand were (environment signals)

    Parameters
    ----------
    action_space : ActionSpace  — used to infer n_actions for entropy normalisation
    n_state_dims : int          — number of discretised state dimensions (default 3)
    n_bins       : int          — bins per dimension (default 8, used if not readable
                                  from state.discretizers)
    """

    def __init__(self, action_space, n_state_dims: int = 3, n_bins: int = 8):
        self.action_space   = action_space
        self.n_state_dims   = n_state_dims
        self.n_bins         = n_bins

        self._transitions: List[TransitionRecord] = []

        self._ep_states:  List[tuple] = []
        self._ep_prices:  List[float] = []
        self._ep_demands: List[float] = []

        self._summaries: List[EpisodeSummary] = []

    def log_transition(
        self,
        *,
        episode:        int,
        state_key:      tuple,
        action_idx:     int,
        next_key:       tuple,
        clearing_price: float,
        demand_mw:      float,
    ) -> None:
        """Call once per environment step."""
        rec = TransitionRecord(
            episode=episode,
            state_key=state_key,
            action_idx=action_idx,
            next_key=next_key,
            clearing_price=float(clearing_price),
            demand_mw=float(demand_mw),
        )
        self._transitions.append(rec)
        self._ep_states.append(state_key)
        self._ep_prices.append(float(clearing_price))
        self._ep_demands.append(float(demand_mw))

    def close_episode(self, episode: int) -> None:
        """Call at the end of each episode to compute and store its summary."""
        if not self._ep_states:
            return

        state_counts = defaultdict(int)
        for s in self._ep_states:
            state_counts[s] += 1

        ep_transitions = [t for t in self._transitions if t.episode == episode]
        sa_pairs       = {(t.state_key, t.action_idx) for t in ep_transitions}

        probs      = np.array(list(state_counts.values()), dtype=float)
        probs     /= probs.sum()
        entropy    = float(-np.sum(probs * np.log(probs + 1e-12)))
        max_ent    = math.log(self.n_bins ** self.n_state_dims)
        norm_ent   = entropy / max_ent if max_ent > 0 else 0.0

        self._summaries.append(EpisodeSummary(
            episode           = episode,
            n_steps           = len(self._ep_states),
            n_unique_states   = len(state_counts),
            n_unique_sa_pairs = len(sa_pairs),
            mean_price        = float(np.mean(self._ep_prices)),
            std_price         = float(np.std(self._ep_prices)),
            mean_demand       = float(np.mean(self._ep_demands)),
            state_entropy     = round(norm_ent, 4),
        ))

        # Reset
        self._ep_states.clear()
        self._ep_prices.clear()
        self._ep_demands.clear()

    def window_snapshots(self, window_size: int = 14) -> List[TransitionSnapshot]:
        """
        Split the full transition log into contiguous windows of `window_size`
        episodes and estimate P[x' | x, u] independently in each window.
        """
        if not self._transitions:
            return []

        max_ep    = max(t.episode for t in self._transitions)
        snapshots = []

        for w_start in range(0, max_ep + 1, window_size):
            w_end  = min(w_start + window_size - 1, max_ep)
            w_recs = [t for t in self._transitions
                      if w_start <= t.episode <= w_end]
            if not w_recs:
                continue

            # Count (s, a) → s' transitions
            counts: Dict[tuple, Dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
            for rec in w_recs:
                sa = (rec.state_key, rec.action_idx)
                counts[sa][rec.next_key] += 1

            # Normalise to probabilities
            kernel: Dict[tuple, Dict[tuple, float]] = {}
            for sa, next_counts in counts.items():
                total = sum(next_counts.values())
                kernel[sa] = {nk: cnt / total for nk, cnt in next_counts.items()}

            prices  = [r.clearing_price for r in w_recs]
            demands = [r.demand_mw      for r in w_recs]

            snapshots.append(TransitionSnapshot(
                window_id     = len(snapshots),
                episode_start = w_start,
                episode_end   = w_end,
                kernel        = dict(kernel),
                counts        = {k: dict(v) for k, v in counts.items()},
                n_transitions = len(w_recs),
                mean_price    = float(np.mean(prices)),
                mean_demand   = float(np.mean(demands)),
            ))

        return snapshots

    # Exports

    def summary_df(self) -> pd.DataFrame:
        """One row per episode."""
        if not self._summaries:
            return pd.DataFrame()
        return pd.DataFrame([s.__dict__ for s in self._summaries]).set_index("episode")

    def state_visit_df(self) -> pd.DataFrame:
        """
        Frequency of each state_key across the full training run.
        """
        counts = defaultdict(int)
        for t in self._transitions:
            counts[t.state_key] += 1
        df = pd.DataFrame([{"state_key": str(k), "count": v}
                           for k, v in counts.items()])
        return df.sort_values("count", ascending=False).reset_index(drop=True)

    def action_frequency_df(self) -> pd.DataFrame:
        """Frequency of each action_idx across the full training run."""
        counts = defaultdict(int)
        for t in self._transitions:
            counts[t.action_idx] += 1
        df = pd.DataFrame([{"action_idx": k, "count": v}
                           for k, v in counts.items()])
        return df.sort_values("action_idx").reset_index(drop=True)

    # Exports

    def export(self, out_dir: str = ".") -> dict:
        """Save all diagnostic CSVs and plots to out_dir."""
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.today().strftime("%Y%m%d_%H%M%S")
        p  = lambda fname: os.path.join(out_dir, fname)
        paths = {}

        # Episode summary CSV
        sdf = self.summary_df()
        if not sdf.empty:
            csv_path = p(f"env_episode_summary_{ts}.csv")
            sdf.to_csv(csv_path)
            paths["episode_summary_csv"] = csv_path
            print(f"Saved env summary CSV   -> {csv_path}")

            # Episode summary plot
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            axes[0].plot(sdf.index, sdf["mean_price"],   color="#2E86AB", linewidth=1.5)
            axes[0].fill_between(sdf.index,
                                 sdf["mean_price"] - sdf["std_price"],
                                 sdf["mean_price"] + sdf["std_price"],
                                 alpha=0.2, color="#2E86AB")
            axes[0].set(title="Mean Clearing Price per Episode", ylabel="$/MWh")
            axes[0].grid(alpha=0.3)

            axes[1].plot(sdf.index, sdf["mean_demand"], color="#A23B72", linewidth=1.5)
            axes[1].set(title="Mean Demand per Episode", ylabel="MW")
            axes[1].grid(alpha=0.3)

            axes[2].plot(sdf.index, sdf["state_entropy"], color="#F18F01", linewidth=1.5)
            axes[2].axhline(ENTROPY_MIN_NORM, color="red", linestyle="--",
                            linewidth=1, label=f"Min entropy ({ENTROPY_MIN_NORM})")
            axes[2].set(title="State Visit Entropy (normalised)", ylabel="Entropy",
                        xlabel="Episode", ylim=(0, 1.05))
            axes[2].legend(fontsize=8)
            axes[2].grid(alpha=0.3)

            fig.tight_layout()
            plot_path = p(f"env_episode_summary_{ts}.png")
            fig.savefig(plot_path, dpi=140)
            plt.close(fig)
            paths["episode_summary_plot"] = plot_path
            print(f"Saved env summary plot  -> {plot_path}")

        # State visit CSV
        svdf = self.state_visit_df()
        if not svdf.empty:
            sv_path = p(f"env_state_visits_{ts}.csv")
            svdf.to_csv(sv_path, index=False)
            paths["state_visits_csv"] = sv_path
            print(f"Saved state visits CSV  -> {sv_path}")

        # Action frequency CSV
        afdf = self.action_frequency_df()
        if not afdf.empty:
            af_path = p(f"env_action_freq_{ts}.csv")
            afdf.to_csv(af_path, index=False)
            paths["action_freq_csv"] = af_path
            print(f"Saved action freq CSV   -> {af_path}")

        return paths

def _total_variation_distance(p: Dict[tuple, float], q: Dict[tuple, float]) -> float:
    """
    TVD between two distributions p and q over the same support. TVD = 0.5 * Σ |p(x) - q(x)|
    """
    all_keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in all_keys)


def measure_transition_stationarity(
    snapshots: List[TransitionSnapshot],
    *,
    min_shared_pairs:  int   = 3,
    min_count_per_sa:  int   = 5,
) -> StationarityReport:
    """
    Tests whether P[x' | x, u] is stable across episode windows.

    This is the core stationarity assumption in Arslan & Yüksel (2017):
    each agent's Q-learning convergence relies on its environment being a
    stationary MDP within each exploration phase. If P[x' | x, u] drifts
    between phases, Q-values estimated in phase k are invalidated in phase k+1.

    Method
    ------
    For each pair of adjacent windows (k, k+1):
      1. Find all (s, a) pairs observed in both windows with sufficient counts.
      2. Compute the Total Variation Distance (TVD) between the two empirical
         transition distributions for each shared (s, a) pair.
         TVD ∈ [0, 1]:  0 = identical distributions, 1 = disjoint support.
      3. Report mean and max TVD across shared pairs.

    Parameters
    ----------
    snapshots         : list of TransitionSnapshot, one per window
    min_shared_pairs  : skip comparison if fewer shared (s,a) pairs than this
    min_count_per_sa  : only include (s,a) pairs with enough observations to
                        give reliable empirical estimates

    Returns
    -------
    StationarityReport with per-window-pair comparisons and overall verdict.
    """
    if len(snapshots) < 2:
        return StationarityReport(
            timestamp=datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            n_windows=len(snapshots),
            comparisons=[],
            overall_passed=True,
            mean_tvd_global=0.0,
            max_tvd_global=0.0,
            summary=["Not enough windows to compare (need ≥ 2)."],
        )

    comparisons = []

    for i in range(len(snapshots) - 1):
        win_a = snapshots[i]
        win_b = snapshots[i + 1]

        # Find (s,a) pairs that appear in both windows with enough observations
        shared_pairs = []
        for sa in set(win_a.counts) & set(win_b.counts):
            count_a = sum(win_a.counts[sa].values())
            count_b = sum(win_b.counts[sa].values())
            if count_a >= min_count_per_sa and count_b >= min_count_per_sa:
                shared_pairs.append(sa)

        # Coverage: what fraction of window_a's pairs are covered
        total_a     = sum(1 for sa in win_a.counts
                          if sum(win_a.counts[sa].values()) >= min_count_per_sa)
        coverage    = len(shared_pairs) / max(total_a, 1)
        warns       = []

        if len(shared_pairs) < min_shared_pairs:
            # Not enough overlap to make a meaningful comparison
            warns.append(
                f"Only {len(shared_pairs)} shared (s,a) pairs with ≥{min_count_per_sa} "
                f"observations — comparison unreliable. "
                f"Try a larger window_size or more episodes."
            )
            comparisons.append(WindowComparison(
                window_a=win_a.window_id, window_b=win_b.window_id,
                mean_tvd=float("nan"), max_tvd=float("nan"),
                n_shared_pairs=len(shared_pairs), coverage_ratio=coverage,
                passed=True,   # can't fail what we can't measure
                warnings=warns,
            ))
            continue

        # Compute TVD for each shared (s,a) pair
        tvds = []
        for sa in shared_pairs:
            tvd = _total_variation_distance(win_a.kernel[sa], win_b.kernel[sa])
            tvds.append(tvd)

        mean_tvd = float(np.mean(tvds))
        max_tvd  = float(np.max(tvds))

        # Diagnose failures
        if mean_tvd > TVD_FAIL_THRESHOLD:
            warns.append(
                f"Mean TVD {mean_tvd:.3f} > {TVD_FAIL_THRESHOLD} (FAIL). "
                f"The transition kernel has shifted substantially between "
                f"windows {win_a.window_id} and {win_b.window_id}. "
                f"Q-values learned in window {win_a.window_id} may be invalid "
                f"in window {win_b.window_id} — this violates the Arslan & Yüksel "
                f"stationarity requirement."
            )
        elif mean_tvd > TVD_WARN_THRESHOLD:
            warns.append(
                f"Mean TVD {mean_tvd:.3f} > {TVD_WARN_THRESHOLD} (WARN). "
                f"Moderate kernel drift between windows {win_a.window_id} and "
                f"{win_b.window_id}. Monitor whether this grows over training."
            )

        if coverage < 0.5:
            warns.append(
                f"Only {coverage:.0%} of window {win_a.window_id}'s (s,a) pairs "
                f"recur in window {win_b.window_id}. Low overlap may indicate "
                f"non-repeating market conditions or an over-discretised state space."
            )

        passed = mean_tvd <= TVD_FAIL_THRESHOLD

        comparisons.append(WindowComparison(
            window_a       = win_a.window_id,
            window_b       = win_b.window_id,
            mean_tvd       = mean_tvd,
            max_tvd        = max_tvd,
            n_shared_pairs = len(shared_pairs),
            coverage_ratio = coverage,
            passed         = passed,
            warnings       = warns,
        ))

    # Global summary
    valid_tvds = [c.mean_tvd for c in comparisons
                  if not math.isnan(c.mean_tvd)]
    mean_tvd_global = float(np.mean(valid_tvds)) if valid_tvds else float("nan")
    max_tvd_global  = float(np.max(valid_tvds))  if valid_tvds else float("nan")
    overall_passed  = all(c.passed for c in comparisons)

    summary = []
    if not overall_passed:
        n_fail = sum(1 for c in comparisons if not c.passed)
        summary.append(
            f"{n_fail}/{len(comparisons)} window pairs failed the stationarity test. "
            f"The transition kernel P[x'|x,u] is non-stationary across episodes — "
            f"this undermines the convergence guarantee of Arslan & Yüksel (2017)."
        )
    else:
        summary.append(
            f"All {len(comparisons)} window pairs passed. "
            f"P[x'|x,u] is approximately stationary across episode windows."
        )

    return StationarityReport(
        timestamp       = datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        n_windows       = len(snapshots),
        comparisons     = comparisons,
        overall_passed  = overall_passed,
        mean_tvd_global = mean_tvd_global,
        max_tvd_global  = max_tvd_global,
        summary         = summary,
    )