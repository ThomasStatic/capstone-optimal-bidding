from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from empirical_game.graph_analysis import GraphAnalysisResult


def sink_quality_summary(
    *,
    sink_ids: list[str],
    deviation_df: pd.DataFrame,
    gap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Diagnostics per sink: outgoing deviations, largest gain, max BR gap."""
    rows: list[dict[str, Any]] = []

    for sink_id in sink_ids:
        out_dev = deviation_df[deviation_df["base_profile_id"] == sink_id].copy()
        profitable = out_dev[out_dev["is_profitable"]]
        sink_gaps = gap_df[gap_df["profile_id"] == sink_id]

        rows.append(
            {
                "sink_profile_id": sink_id,
                "n_outgoing_deviations": int(len(out_dev)),
                "n_profitable_deviations": int(len(profitable)),
                "largest_profitable_gain": float(profitable["gain_mean"].max()) if not profitable.empty else 0.0,
                "max_best_response_gap": float(sink_gaps["best_response_gap"].max()) if not sink_gaps.empty else 0.0,
            }
        )

    return pd.DataFrame(rows)


def local_robustness_around_profile(
    *,
    target_profile_id: str,
    deviation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Filter to perturbation-only unilateral deviations around a target profile."""
    local = deviation_df[
        (deviation_df["base_profile_id"] == target_profile_id)
        & (deviation_df["to_policy_kind"] == "perturbation")
    ].copy()

    if local.empty:
        return pd.DataFrame(
            columns=[
                "base_profile_id",
                "agent_id",
                "from_policy_id",
                "to_policy_id",
                "gain_mean",
                "gain_lcb",
                "is_profitable",
            ]
        )

    return local[
        [
            "base_profile_id",
            "agent_id",
            "from_policy_id",
            "to_policy_id",
            "gain_mean",
            "gain_lcb",
            "is_profitable",
        ]
    ].reset_index(drop=True)


def graph_summary_dataframe(result: GraphAnalysisResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "weakly_acyclic": result.weakly_acyclic,
                "n_nodes": result.n_nodes,
                "n_edges": result.n_edges,
                "n_sinks": len(result.sinks),
                "n_problematic_nodes": len(result.problematic_nodes),
            }
        ]
    )


def export_tables(
    *,
    output_dir: str,
    payoff_df: pd.DataFrame,
    deviation_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    sink_df: pd.DataFrame,
    local_robustness_df: pd.DataFrame,
    graph_result: GraphAnalysisResult,
) -> dict[str, str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    paths = {
        "payoff": str(Path(output_dir) / "payoff_table.csv"),
        "deviations": str(Path(output_dir) / "deviation_table.csv"),
        "edges": str(Path(output_dir) / "best_reply_edges.csv"),
        "gaps": str(Path(output_dir) / "best_response_gaps.csv"),
        "sink_quality": str(Path(output_dir) / "sink_quality.csv"),
        "local_robustness": str(Path(output_dir) / "local_robustness.csv"),
        "graph_summary": str(Path(output_dir) / "graph_summary.csv"),
        "graph_reachability": str(Path(output_dir) / "graph_reachable_sinks.csv"),
        "graph_scc": str(Path(output_dir) / "graph_sccs.csv"),
    }

    payoff_df.to_csv(paths["payoff"], index=False)
    deviation_df.to_csv(paths["deviations"], index=False)
    edge_df.to_csv(paths["edges"], index=False)
    gap_df.to_csv(paths["gaps"], index=False)
    sink_df.to_csv(paths["sink_quality"], index=False)
    local_robustness_df.to_csv(paths["local_robustness"], index=False)
    graph_summary_dataframe(graph_result).to_csv(paths["graph_summary"], index=False)

    reach_rows = [
        {"profile_id": k, "reachable_sinks": "|".join(v)}
        for k, v in graph_result.reachable_sinks_by_node.items()
    ]
    pd.DataFrame(reach_rows).to_csv(paths["graph_reachability"], index=False)

    scc_rows = [{"scc_index": i, "nodes": "|".join(nodes)} for i, nodes in enumerate(graph_result.sccs)]
    pd.DataFrame(scc_rows).to_csv(paths["graph_scc"], index=False)

    return paths
