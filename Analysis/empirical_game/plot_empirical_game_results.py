from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"


def _abbr_token(token: str) -> str:
    token = token.strip()
    if "cost_plus" in token:
        return "CP"
    if "hist_quantile" in token:
        return "HQ"
    if "learned" in token and "__" not in token:
        return "L"

    if "__" in token:
        base, mut = token.split("__", 1)
        base_code = "L" if "learned" in base else "X"
        mut = mut.replace("_agent_0", "").replace("_agent_1", "")
        mut = mut.replace("price_up", "P+").replace("price_down", "P-")
        mut = mut.replace("quantity_up", "Q+").replace("quantity_down", "Q-")
        mut = mut.replace("_1", "")
        return f"{base_code}:{mut}"

    return token[:12]


def _abbr_profile(profile_id: str) -> str:
    parts = profile_id.split("|")
    short_parts = []
    for p in parts:
        if ":" in p:
            side, token = p.split(":", 1)
            side = side.upper()
            short_parts.append(f"{side}-{_abbr_token(token)}")
        else:
            short_parts.append(_abbr_token(p))
    return " | ".join(short_parts)


def _plot_graph_overview(summary: pd.DataFrame, sink_quality: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    metrics = {
        "Nodes": int(summary["n_nodes"].iloc[0]),
        "Edges": int(summary["n_edges"].iloc[0]),
        "Sinks": int(summary["n_sinks"].iloc[0]),
        "Problem Nodes": int(summary["n_problematic_nodes"].iloc[0]),
    }
    weakly_acyclic = bool(summary["weakly_acyclic"].iloc[0])
    colors = ["#3a86ff", "#2a9d8f", "#ffb703", "#e76f51"]

    axes[0].bar(metrics.keys(), metrics.values(), color=colors)
    axes[0].set_title("Improvement Graph Size")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(metrics.values()):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    status = "True" if weakly_acyclic else "False"
    subtitle = "All profiles can reach a sink" if weakly_acyclic else "Some profiles cannot reach any sink"
    axes[1].axis("off")
    axes[1].text(0.05, 0.78, f"weakly_acyclic = {status}", fontsize=16, weight="bold")
    axes[1].text(0.05, 0.62, subtitle, fontsize=11)

    if not sink_quality.empty:
        sq = sink_quality.iloc[0]
        text = (
            f"sink_profile_id: {_abbr_profile(str(sq['sink_profile_id']))}\n"
            f"outgoing deviations: {int(sq['n_outgoing_deviations'])}\n"
            f"profitable deviations: {int(sq['n_profitable_deviations'])}\n"
            f"largest profitable gain: {sq['largest_profitable_gain']:.1f}\n"
            f"max best-response gap: {sq['max_best_response_gap']:.1f}"
        )
        axes[1].text(0.05, 0.20, text, fontsize=10, family="monospace")

    fig.suptitle("Weak Acyclicity Diagnostics", fontsize=14, weight="bold")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_best_response_heatmap(best_response_gaps: pd.DataFrame, out_path: Path) -> None:
    gap = best_response_gaps.copy()
    gap["profile_label"] = gap["profile_id"].map(_abbr_profile)

    pivot = (
        gap.pivot_table(index="profile_label", columns="agent_id", values="best_response_gap", aggfunc="mean")
        .sort_values(by=list(gap["agent_id"].unique()), ascending=False)
    )

    vmax = np.nanmax(np.abs(pivot.to_numpy()))
    fig_h = max(6, min(18, 0.3 * len(pivot)))
    fig, ax = plt.subplots(figsize=(10, fig_h), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)

    ax.set_title("Best-Response Gap by Profile and Agent")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Profile")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"agent_{int(c)}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Best-response gap")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_deviation_kind_bars(deviation_table: pd.DataFrame, out_path: Path) -> None:
    dev = deviation_table.copy()
    grouped = (
        dev.groupby(["agent_id", "to_policy_kind"])
        .agg(
            profitable_rate=("is_profitable", "mean"),
            avg_gain_lcb=("gain_lcb", "mean"),
            n=("is_profitable", "size"),
        )
        .reset_index()
    )
    grouped["profitable_rate_pct"] = 100.0 * grouped["profitable_rate"]

    kinds = [k for k in ["baseline", "perturbation", "learned"] if k in grouped["to_policy_kind"].unique()]
    agent_ids = sorted(grouped["agent_id"].unique())
    x = np.arange(len(kinds))
    width = 0.35 if len(agent_ids) > 1 else 0.55

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    palette = ["#264653", "#e76f51", "#2a9d8f", "#457b9d"]

    for idx, agent_id in enumerate(agent_ids):
        data = grouped[grouped["agent_id"] == agent_id].set_index("to_policy_kind").reindex(kinds)
        offset = (idx - (len(agent_ids) - 1) / 2.0) * width
        axes[0].bar(x + offset, data["profitable_rate_pct"], width=width, label=f"agent_{int(agent_id)}", color=palette[idx % len(palette)])
        axes[1].bar(x + offset, data["avg_gain_lcb"], width=width, label=f"agent_{int(agent_id)}", color=palette[idx % len(palette)])

    axes[0].set_title("Profitable Deviation Rate")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(kinds)
    axes[0].set_ylim(0, 100)
    axes[0].legend()

    axes[1].set_title("Average Conservative Gain (LCB)")
    axes[1].set_ylabel("Gain LCB")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(kinds)
    axes[1].axhline(0, color="black", linewidth=1)

    fig.suptitle("Deviation Economics by Target Policy Kind", fontsize=14, weight="bold")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_top_best_reply_edges(best_reply_edges: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    edges = best_reply_edges.nlargest(top_n, "gain_lcb").copy()
    edges["edge_label"] = edges.apply(
        lambda r: f"A{int(r['agent_id'])}: {_abbr_token(str(r['from_policy_id']))} -> {_abbr_token(str(r['to_policy_id']))}",
        axis=1,
    )
    edges = edges.sort_values("gain_lcb", ascending=True)

    fig_h = max(6, 0.35 * len(edges) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h), constrained_layout=True)
    colors = np.where(edges["agent_id"] == 0, "#1d3557", "#e63946")
    ax.barh(edges["edge_label"], edges["gain_lcb"], color=colors)
    ax.set_title(f"Top {top_n} Best-Reply Edges by Conservative Gain")
    ax.set_xlabel("Gain LCB")
    ax.set_ylabel("Deviation")

    legend_handles = [
        plt.Line2D([0], [0], color="#1d3557", lw=8, label="agent_0"),
        plt.Line2D([0], [0], color="#e63946", lw=8, label="agent_1"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_payoff_landscape(payoff_table: pd.DataFrame, sink_profiles: Iterable[str], out_path: Path) -> None:
    pv = (
        payoff_table.pivot_table(index="profile_id", columns="agent_id", values="mean_payoff", aggfunc="mean")
        .rename(columns=lambda c: f"agent_{int(c)}")
        .reset_index()
    )
    if "agent_0" not in pv.columns or "agent_1" not in pv.columns:
        return

    sink_set = set(sink_profiles)
    pv["is_sink"] = pv["profile_id"].isin(sink_set)
    pv["profile_label"] = pv["profile_id"].map(_abbr_profile)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.scatter(
        pv.loc[~pv["is_sink"], "agent_0"],
        pv.loc[~pv["is_sink"], "agent_1"],
        alpha=0.75,
        s=45,
        c="#457b9d",
        label="non-sink profile",
    )
    if pv["is_sink"].any():
        ax.scatter(
            pv.loc[pv["is_sink"], "agent_0"],
            pv.loc[pv["is_sink"], "agent_1"],
            alpha=1.0,
            s=120,
            c="#e63946",
            marker="*",
            label="sink profile",
        )

    ax.set_xlabel("Agent 0 mean payoff")
    ax.set_ylabel("Agent 1 mean payoff")
    ax.set_title("Payoff Landscape Across Strategy Profiles")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_reachable_sink_counts(reachable_sinks: pd.DataFrame, out_path: Path) -> None:
    rs = reachable_sinks.copy()
    counts = rs["reachable_sinks"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    labels = [_abbr_profile(x) for x in counts.index]
    ax.bar(labels, counts.values, color="#2a9d8f")
    ax.set_title("How Many Profiles Reach Each Sink")
    ax.set_xlabel("Sink profile")
    ax.set_ylabel("Number of starting profiles")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right")

    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_all_plots(root: Path = ROOT) -> Dict[str, Path]:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    best_reply_edges = pd.read_csv(root / "best_reply_edges.csv")
    best_response_gaps = pd.read_csv(root / "best_response_gaps.csv")
    deviation_table = pd.read_csv(root / "deviation_table.csv")
    graph_reachable_sinks = pd.read_csv(root / "graph_reachable_sinks.csv")
    graph_summary = pd.read_csv(root / "graph_summary.csv")
    payoff_table = pd.read_csv(root / "payoff_table.csv")
    sink_quality = pd.read_csv(root / "sink_quality.csv")

    output_paths: Dict[str, Path] = {
        "graph_overview": fig_dir / "graph_overview.png",
        "best_response_heatmap": fig_dir / "best_response_gap_heatmap.png",
        "deviation_kind_bars": fig_dir / "deviation_kind_bars.png",
        "top_best_reply_edges": fig_dir / "top_best_reply_edges.png",
        "payoff_landscape": fig_dir / "payoff_landscape.png",
        "reachable_sink_counts": fig_dir / "reachable_sink_counts.png",
    }

    _plot_graph_overview(graph_summary, sink_quality, output_paths["graph_overview"])
    _plot_best_response_heatmap(best_response_gaps, output_paths["best_response_heatmap"])
    _plot_deviation_kind_bars(deviation_table, output_paths["deviation_kind_bars"])
    _plot_top_best_reply_edges(best_reply_edges, output_paths["top_best_reply_edges"])
    _plot_payoff_landscape(
        payoff_table,
        sink_profiles=graph_reachable_sinks["reachable_sinks"].unique(),
        out_path=output_paths["payoff_landscape"],
    )
    _plot_reachable_sink_counts(graph_reachable_sinks, output_paths["reachable_sink_counts"])

    return output_paths


def main() -> None:
    outputs = build_all_plots(ROOT)
    print("Saved figures:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()