from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GraphAnalysisResult:
    weakly_acyclic: bool
    sinks: list[str]
    reachable_sinks_by_node: dict[str, list[str]]
    problematic_nodes: list[str]
    sccs: list[list[str]]
    n_nodes: int
    n_edges: int


def build_adjacency(all_nodes: list[str], edge_df: pd.DataFrame) -> dict[str, list[str]]:
    adj = {n: [] for n in all_nodes}
    if edge_df.empty:
        return adj

    for _, row in edge_df.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        if src in adj:
            adj[src].append(tgt)

    for n in all_nodes:
        adj[n] = sorted(set(adj[n]))

    return adj


def _reverse_adjacency(adjacency: dict[str, list[str]]) -> dict[str, list[str]]:
    rev = {k: [] for k in adjacency.keys()}
    for src, targets in adjacency.items():
        for tgt in targets:
            rev[tgt].append(src)
    return rev


def _dfs(start: str, adjacency: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adjacency.get(cur, []))
    return seen


def strongly_connected_components(adjacency: dict[str, list[str]]) -> list[list[str]]:
    """Kosaraju SCC decomposition."""
    visited: set[str] = set()
    order: list[str] = []

    def dfs1(node: str) -> None:
        visited.add(node)
        for nxt in adjacency.get(node, []):
            if nxt not in visited:
                dfs1(nxt)
        order.append(node)

    for node in adjacency.keys():
        if node not in visited:
            dfs1(node)

    rev = _reverse_adjacency(adjacency)
    visited.clear()
    sccs: list[list[str]] = []

    def dfs2(node: str, comp: list[str]) -> None:
        visited.add(node)
        comp.append(node)
        for nxt in rev.get(node, []):
            if nxt not in visited:
                dfs2(nxt, comp)

    for node in reversed(order):
        if node not in visited:
            comp: list[str] = []
            dfs2(node, comp)
            sccs.append(sorted(comp))

    return sccs


def analyze_weak_acyclicity(
    *,
    all_nodes: list[str],
    edge_df: pd.DataFrame,
    sinks: list[str],
) -> GraphAnalysisResult:
    adj = build_adjacency(all_nodes, edge_df)

    reachable_sinks_by_node: dict[str, list[str]] = {}
    for node in all_nodes:
        reachable = _dfs(node, adj)
        reachable_sinks_by_node[node] = sorted([s for s in sinks if s in reachable])

    problematic = sorted([n for n, rs in reachable_sinks_by_node.items() if not rs])

    return GraphAnalysisResult(
        weakly_acyclic=(len(problematic) == 0),
        sinks=sorted(sinks),
        reachable_sinks_by_node=reachable_sinks_by_node,
        problematic_nodes=problematic,
        sccs=strongly_connected_components(adj),
        n_nodes=len(all_nodes),
        n_edges=int(0 if edge_df.empty else len(edge_df)),
    )
