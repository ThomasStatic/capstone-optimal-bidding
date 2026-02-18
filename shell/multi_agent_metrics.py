from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

AGENT_COLORS = ["#2E86AB", "#E84855", "#3BB273", "#F18F01"]

@dataclasses.dataclass
class StepRecord:
    episode: int
    step: int
    timestamp: pd.Timestamp
    action_indices: List[int]
    rewards: List[float]
    clearing_price: float
    demand_mw: float
    rho: float
    clipped: List[bool]

class MetricsTracker:
    """
    Collects per-step data during training and exposes aggregated DataFrames
    for reporting and PDF export.
    """

    def __init__(self, n_agents: int, action_space=None):
        self.n_agents = n_agents
        self.action_space = action_space
        self._steps: List[StepRecord] = []
        self._episode_summaries: List[dict] = []

        self._ep_rewards: dict[int, List[float]] = defaultdict(list)

    def log_step(
        self,
        *,
        episode: int,
        step: int,
        timestamp: pd.Timestamp,
        action_indices: List[int],
        rewards: List[float],
        clearing_price: float,
        demand_mw: float,
        rho: float,
        clip_infos: Optional[List[dict]] = None,
    ) -> None:
        clipped = []
        if clip_infos:
            clipped = [bool(ci.get("clipped", False)) for ci in clip_infos]
        else:
            clipped = [False] * self.n_agents

        rec = StepRecord(
            episode=episode,
            step=step,
            timestamp=timestamp,
            action_indices=list(action_indices),
            rewards=list(rewards),
            clearing_price=float(clearing_price),
            demand_mw=float(demand_mw),
            rho=float(rho),
            clipped=clipped,
        )
        self._steps.append(rec)

        for i, r in enumerate(rewards):
            self._ep_rewards[i].append(r)

    def close_episode(self, episode: int) -> None:
        """Cleanly generate episode summary, then reset"""
        ep_steps = [s for s in self._steps if s.episode == episode]
        if not ep_steps:
            return

        prices = [s.clearing_price for s in ep_steps]
        demands = [s.demand_mw for s in ep_steps]
        rhos = [s.rho for s in ep_steps]

        summary: dict = {
            "episode": episode,
            "n_steps": len(ep_steps),
            "mean_clearing_price": float(np.mean(prices)),
            "std_clearing_price": float(np.std(prices)),
            "min_clearing_price": float(np.min(prices)),
            "max_clearing_price": float(np.max(prices)),
            "mean_demand_mw": float(np.mean(demands)),
            "mean_rho": float(np.mean(rhos)),
        }

        # Loop for per agent statistics
        for i in range(self.n_agents):
            agent_rewards = [s.rewards[i] for s in ep_steps if i < len(s.rewards)]
            agent_actions = [s.action_indices[i] for s in ep_steps if i < len(s.action_indices)]
            agent_clipped = [s.clipped[i] for s in ep_steps if i < len(s.clipped)]

            summary[f"agent_{i}_total_reward"] = float(np.sum(agent_rewards))
            summary[f"agent_{i}_mean_reward"] = float(np.mean(agent_rewards))
            summary[f"agent_{i}_std_reward"] = float(np.std(agent_rewards))
            summary[f"agent_{i}_clip_rate"] = float(np.mean(agent_clipped))
            # summary[f"agent_{i}_mean_action_idx"] = float(np.mean(agent_actions))
            summary[f"agent_{i}_action_entropy"] = float(self._entropy(agent_actions))

            # Produce bid-action space distributions
            if self.action_space is not None:
                try:
                    bid_prices = []
                    bid_qtys = []
                    for aidx in agent_actions:
                        bid_p, bid_q = self.action_space.decode_action(aidx)
                        bid_prices.append(bid_p)
                        bid_qtys.append(bid_q)
                    summary[f"agent_{i}_mean_bid_price"] = float(np.mean(bid_prices))
                    summary[f"agent_{i}_mean_bid_qty_mw"] = float(np.mean(bid_qtys))
                except Exception:
                    pass

        self._episode_summaries.append(summary)

        # Reset episode metrics
        self._ep_rewards.clear()
        
    def step_df(self) -> pd.DataFrame:
        """Produces flat dataframe per step"""
        if not self._steps:
            return pd.DataFrame()

        rows = []
        for s in self._steps:
            base = {
                "episode": s.episode,
                "step": s.step,
                "timestamp": s.timestamp,
                "clearing_price": s.clearing_price,
                "demand_mw": s.demand_mw,
                "rho": s.rho,
            }
            for i in range(self.n_agents):
                base[f"agent_{i}_reward"] = s.rewards[i] if i < len(s.rewards) else np.nan
                base[f"agent_{i}_action_idx"] = s.action_indices[i] if i < len(s.action_indices) else np.nan
                base[f"agent_{i}_clipped"] = s.clipped[i] if i < len(s.clipped) else False
            rows.append(base)

        return pd.DataFrame(rows)

    def episode_summary_df(self) -> pd.DataFrame:
        if not self._episode_summaries:
            return pd.DataFrame()
        return pd.DataFrame(self._episode_summaries).set_index("episode")

    def agent_kpi_df(self) -> pd.DataFrame:
        """Per-agent KPI table, producing mean over all episodes."""
        edf = self.episode_summary_df()
        if edf.empty:
            return pd.DataFrame()

        records = []
        for i in range(self.n_agents):
            cols = {k: v for k, v in edf.items() if k.startswith(f"agent_{i}_")}
            if not cols:
                continue
            row = {"agent": i}
            row.update({k.replace(f"agent_{i}_", ""): float(np.mean(v)) for k, v in cols.items()})
            records.append(row)

        return pd.DataFrame(records).set_index("agent")

    def bid_distribution_df(self) -> pd.DataFrame:
        """
        Returns action-index frequency per agent across all steps.
        """
        sdf = self.step_df()
        if sdf.empty:
            return pd.DataFrame()

        frames = []
        for i in range(self.n_agents):
            col = f"agent_{i}_action_idx"
            if col not in sdf.columns:
                continue
            counts = sdf[col].dropna().astype(int).value_counts().sort_index()
            df = counts.rename("count").reset_index()
            df.columns = ["action_idx", "count"]
            df["agent"] = i
            frames.append(df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    ## Function for competition metrics
    def competition_table(self) -> pd.DataFrame:
        """One row per agent with competition KPIs."""
        edf = self.episode_summary_df()
        records = []

        for i in range(self.n_agents):
            reward_col = f"agent_{i}_total_reward"
            if reward_col not in edf.columns:
                continue

            # Total rewards across all agents per episode
            all_reward_cols = [f"agent_{j}_total_reward" for j in range(self.n_agents)
                            if f"agent_{j}_total_reward" in edf.columns]
            total_rewards = edf[all_reward_cols].sum(axis=1)

            profit_share = (edf[reward_col] / total_rewards.replace(0, float("nan"))).mean()
            win_rate     = (edf[reward_col] == edf[all_reward_cols].max(axis=1)).mean()
            price_impact = edf[reward_col].corr(edf["mean_clearing_price"])

            spread = None
            if self.action_space is not None:
                try:
                    mean_idx = edf[f"agent_{i}_mean_action_idx"].mean()
                    bid_price, _ = self.action_space.decode_to_values(int(round(mean_idx)))
                    spread = round(bid_price - edf["mean_clearing_price"].mean(), 2)
                except Exception:
                    pass
            records[-1]["market_spread"] = spread

            records.append({
                "agent":        i,
                "profit_share": round(profit_share * 100, 2), 
                "win_rate":     round(win_rate * 100, 2),   
                "price_impact": round(price_impact, 3),
            })

        return pd.DataFrame(records).set_index("agent")

# Helpers
    @staticmethod
    def _entropy(actions: List[int]) -> float:
        """Entropy on action distribution. 
        Basically determines how spread out an agent's bids are across action indices"""
        if not actions:
            return 0.0
        _, counts = np.unique(actions, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def summary_stats(self) -> dict:
        """Summary printed to console."""
        edf = self.episode_summary_df()
        if edf.empty:
            return {}
        out = {
            "total_episodes": len(edf),
            "total_steps": len(self._steps),
            "mean_clearing_price": float(edf["mean_clearing_price"].mean()),
            "mean_demand_mw": float(edf["mean_demand_mw"].mean()),
        }
        for i in range(self.n_agents):
            col = f"agent_{i}_total_reward"
            if col in edf.columns:
                out[f"agent_{i}_mean_total_reward"] = float(edf[col].mean())
        return out
    
# Export Metrics for multi-agent simulation
def export_episode_csv(metrics, path="episode_metrics.csv"):
    """
    One row per episode with the following columns: episode, mean_clearing_price,std_clearing_price, 
    mean_demand_mw, mean_rho, plus per-agent, total_reward and mean_reward.
    """
    df = metrics.episode_summary_df()
    df.to_csv(path)
    print(f"Saved episode CSV  -> {path}")
    return path


def export_agent_csv(metrics, path="agent_kpis.csv"):
    """
    One row per agent with the following columnsolumns: mean_reward, std_reward, clip_rate, action_entropy.
    These are all averaged over the episodes.
    """
    df = metrics.agent_kpi_df()
    df.to_csv(path)
    print(f"Saved agent CSV    -> {path}")
    return path

def export_competition_csv(metrics, path="competition_table.csv"):
    """Comparison table across agents: profit share, win rate, price impact, market spread."""
    df = metrics.competition_table()
    df.to_csv(path)
    print(f"Saved competition CSV -> {path}")
    return path

def plot_rewards(metrics, path="episode_rewards.png"):
    edf = metrics.episode_summary_df()
    fig, ax = plt.subplots(figsize=(9, 4))

    for i in range(metrics.n_agents):
        col = f"agent_{i}_total_reward"
        if col in edf.columns:
            ax.plot(edf.index, edf[col], label=f"Agent {i}",
                    color=AGENT_COLORS[i % len(AGENT_COLORS)], linewidth=2)

    ax.set(title="Reward per Episode", xlabel="Episode", ylabel="Total Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved rewards plot -> {path}")
    return path


def plot_clearing_price(metrics, path="clearing_price.png"):
    edf = metrics.episode_summary_df()
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(edf.index, edf["mean_clearing_price"],
            color="#2E86AB", linewidth=2, label="Mean price")
    # ax.fill_between(
    #     edf.index,
    #     edf["mean_clearing_price"] - edf["std_clearing_price"],
    #     edf["mean_clearing_price"] + edf["std_clearing_price"],
    #     alpha=0.2, color="#2E86AB", label="±1 std",
    # )

    ax.set(title="Market Clearing Price per Episode",
           xlabel="Episode", ylabel="Price ($/MWh)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved price plot   -> {path}")
    return path


def plot_bid_distributions(metrics, path="bid_distributions.png"):
    bdf = metrics.bid_distribution_df()
    n = metrics.n_agents
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i in range(n):
        ax = axes[0][i]
        sub = bdf[bdf["agent"] == i] if not bdf.empty else None
        if sub is not None and not sub.empty:
            ax.bar(sub["action_idx"], sub["count"],
                   color=AGENT_COLORS[i % len(AGENT_COLORS)], alpha=0.85)
        ax.set(title=f"Agent {i} — Bid Distribution",
               xlabel="Action Index", ylabel="Count")
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved bid dist plot-> {path}")
    return path

def export_multi_agent_metrics(metrics, out_dir="Analysis\Metrics\Multi_Agent."):
    """
    Run all exports (2 CSVs + 3 charts) into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    p = lambda fname: os.path.join(out_dir, fname)

    current_time = datetime.today().strftime("%Y%M%d_%H%M%S")

    return {
        "episode_csv":        export_episode_csv(metrics,        p(f"episode_metrics_{current_time}.csv")),
        "agent_csv":          export_agent_csv(metrics,          p(f"agent_kpis_{current_time}.csv")),
        "rewards_plot":       plot_rewards(metrics,              p(f"episode_rewards_{current_time}.png")),
        "clearing_price_plot":plot_clearing_price(metrics,       p(f"clearing_price_{current_time}.png")),
        "bid_dist_plot":      plot_bid_distributions(metrics,    p(f"bid_distributions_{current_time}.png")),
        "competition_csv": export_competition_csv(metrics, p(f"competition_table_{current_time}.csv")),
    }