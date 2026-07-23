"""Exploratory diagnostics for the provenance-complete forward graph.

This module does not alter the frozen primary gate or generate TABI candidates.
It makes two reviewer-requested checks auditable: how the global size gate
behaves on a fragmented graph, and whether the forward run's temporal null is
also observed when activity is normalized by unique supporting papers rather
than extracted relation events.  The legacy diagnostic cannot support the
second check because its paper identifiers are incomplete.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _configure_run_id_from_cli() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id")
    args, _ = parser.parse_known_args()
    if args.run_id:
        os.environ["KG_TABI_RUN_ID"] = args.run_id


_configure_run_id_from_cli()

import networkx as nx

from src import config
from src.graph_analysis import (
    _benjamini_hochberg,
    _mann_kendall,
    _sen_slope,
    build_graph,
    community_cut_edge_metrics,
    louvain_partition,
)
from src.provenance import file_sha256, utc_now_iso


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def structural_gate_diagnostic(graph: nx.DiGraph) -> dict[str, Any]:
    """Compare declared size gates without selecting a replacement gate."""
    projection = graph.to_undirected()
    partition = louvain_partition(graph)
    metrics = community_cut_edge_metrics(projection, partition)
    component_size_by_node = {
        node: len(component)
        for component in nx.connected_components(projection)
        for node in component
    }
    rows = []
    for metric in metrics:
        community_nodes = [node for node, community_id in partition.items()
                           if str(community_id) == str(metric["community_id"])]
        component_size = component_size_by_node[community_nodes[0]] if community_nodes else 0
        rows.append({
            **metric,
            "component_node_count": component_size,
            "component_relative_size_ratio": (
                metric["node_count"] / component_size if component_size else 0.0
            ),
            "passes_bridge_gate": metric["cut_edge_fraction"] <= config.LOUVAIN_MAX_BRIDGE_RATIO,
        })

    absolute_minimum = 10
    profiles = {
        "frozen_global_5_percent": lambda row: row["size_ratio"] >= config.LOUVAIN_MIN_SIZE_RATIO,
        "exploratory_absolute_10_nodes": lambda row: row["node_count"] >= absolute_minimum,
        "exploratory_component_relative_5_percent_plus_10_nodes": lambda row: (
            row["node_count"] >= max(absolute_minimum, math.ceil(0.05 * row["component_node_count"]))
        ),
    }
    results = {}
    for name, size_rule in profiles.items():
        passed = [row["community_id"] for row in rows if size_rule(row) and row["passes_bridge_gate"]]
        results[name] = {
            "size_rule": (
                "node_count / all_graph_nodes >= 0.05" if name == "frozen_global_5_percent"
                else "node_count >= 10" if name == "exploratory_absolute_10_nodes"
                else "node_count >= max(10, ceil(0.05 * connected_component_node_count))"
            ),
            "bridge_rule": f"cut_edge_fraction <= {config.LOUVAIN_MAX_BRIDGE_RATIO}",
            "communities_passing_both": len(passed),
            "community_ids_passing_both": passed,
        }

    largest = max(rows, key=lambda row: (row["node_count"], str(row["community_id"])), default=None)
    return {
        "scope": (
            "Exploratory gate diagnostic only. It does not calibrate a replacement threshold, "
            "evaluate scientific relevance, or authorize TABI generation under a relaxed gate."
        ),
        "graph": {
            "nodes": graph.number_of_nodes(),
            "directed_edges": graph.number_of_edges(),
            "connected_components_in_undirected_projection": nx.number_connected_components(projection),
            "louvain_communities": len(rows),
        },
        "frozen_gate": {
            "minimum_global_size_ratio": config.LOUVAIN_MIN_SIZE_RATIO,
            "minimum_absolute_nodes_implied": math.ceil(
                config.LOUVAIN_MIN_SIZE_RATIO * graph.number_of_nodes()
            ),
            "maximum_cut_edge_fraction": config.LOUVAIN_MAX_BRIDGE_RATIO,
        },
        "largest_community": largest,
        "communities_passing_bridge_gate_before_any_size_rule": sum(
            row["passes_bridge_gate"] for row in rows
        ),
        "profiles": results,
        "community_rows": sorted(rows, key=lambda row: (-row["node_count"], str(row["community_id"]))),
    }


def paper_normalized_temporal_diagnostic(
    graph: nx.DiGraph, *, cutoff_year: int
) -> dict[str, Any]:
    """Run the frozen temporal screen using unique paper IDs as the exposure."""
    node_papers: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    corpus_papers: dict[int, set[str]] = defaultdict(set)
    missing_paper_id_events = 0
    excluded_after_cutoff = 0
    for source, target, data in graph.edges(data=True):
        for event in data.get("events", []):
            try:
                year = int(event.get("year"))
            except (TypeError, ValueError):
                continue
            if year > cutoff_year:
                excluded_after_cutoff += 1
                continue
            paper_id = str(event.get("paper_id") or "").strip()
            if not paper_id:
                missing_paper_id_events += 1
                continue
            corpus_papers[year].add(paper_id)
            node_papers[source][year].add(paper_id)
            if source != target:
                node_papers[target][year].add(paper_id)

    covered_years = sorted(year for year, papers in corpus_papers.items() if papers)
    report: dict[str, Any] = {
        "scope": (
            "Forward-run exploratory sensitivity. It is unavailable for the legacy graph, "
            "whose event-level paper identifiers are incomplete; it does not validate temporal "
            "research activity or calibrate a new primary screen."
        ),
        "normalization": "Unique supporting papers incident to node / unique retained corpus papers in year.",
        "cutoff_year": cutoff_year,
        "missing_paper_id_events": missing_paper_id_events,
        "events_excluded_after_cutoff": excluded_after_cutoff,
        "coverage": {
            "covered_years": covered_years,
            "unique_papers_by_year": {str(year): len(corpus_papers[year]) for year in covered_years},
        },
        "thresholds_reused_without_recalibration": {
            "minimum_unique_papers": config.TEMPORAL_MIN_EVENTS,
            "minimum_distinct_years": config.TEMPORAL_MIN_DISTINCT_YEARS,
            "minimum_negative_sen_slope": config.TEMPORAL_MIN_NEGATIVE_SLOPE,
            "normalized_decay_threshold": config.TEMPORAL_DECAY_THRESHOLD,
            "fdr_significance_level": config.TEMPORAL_FDR_SIGNIFICANCE_LEVEL,
        },
        "eligible_node_statistics": [],
    }
    if len(covered_years) <= config.TEMPORAL_RECENT_WINDOW_YEARS:
        report["status"] = "insufficient_covered_years_for_historical_comparison"
        return report

    recent_years = covered_years[-config.TEMPORAL_RECENT_WINDOW_YEARS:]
    historical_years = covered_years[:-config.TEMPORAL_RECENT_WINDOW_YEARS]
    candidates = []
    counts = Counter()
    for node in sorted(graph.nodes(), key=lambda value: str(value).casefold()):
        if str(node).strip().lower() in config.GENERIC_TEMPORAL_NODES:
            continue
        counts["non_generic_nodes"] += 1
        papers_by_year = node_papers[node]
        total_papers = len(set().union(*papers_by_year.values())) if papers_by_year else 0
        if total_papers < config.TEMPORAL_MIN_EVENTS:
            continue
        counts["nodes_with_minimum_unique_papers"] += 1
        observed_years = [year for year in covered_years if papers_by_year[year]]
        if len(observed_years) < config.TEMPORAL_MIN_DISTINCT_YEARS:
            continue
        counts["eligible_nodes_tested"] += 1
        annual_share = [len(papers_by_year[year]) / len(corpus_papers[year]) for year in covered_years]
        historical = annual_share[:-config.TEMPORAL_RECENT_WINDOW_YEARS]
        recent = annual_share[-config.TEMPORAL_RECENT_WINDOW_YEARS:]
        historical_mean = sum(historical) / len(historical)
        recent_mean = sum(recent) / len(recent)
        decay = max(0.0, min(1.0, 1.0 - recent_mean / historical_mean)) if historical_mean else 0.0
        z_score, p_value = _mann_kendall(annual_share)
        slope = _sen_slope(covered_years, annual_share)
        candidates.append({
            "node": node,
            "node_type": graph.nodes[node].get("type", "CONCEPT"),
            "unique_supporting_papers": total_papers,
            "observed_paper_years": observed_years,
            "annual_unique_papers": {str(year): len(papers_by_year[year]) for year in covered_years},
            "annual_paper_share": {str(year): value for year, value in zip(covered_years, annual_share)},
            "historical_paper_normalized_mean": historical_mean,
            "recent_paper_normalized_mean": recent_mean,
            "decay_rate": decay,
            "mann_kendall_z": z_score,
            "mann_kendall_p": p_value,
            "sen_slope": slope,
        })

    adjusted = _benjamini_hochberg([row["mann_kendall_p"] for row in candidates])
    signals = []
    for row, q_value in zip(candidates, adjusted):
        row["mann_kendall_q"] = q_value
        row["passes_negative_slope"] = row["sen_slope"] < -config.TEMPORAL_MIN_NEGATIVE_SLOPE
        row["passes_decay"] = row["decay_rate"] >= config.TEMPORAL_DECAY_THRESHOLD
        row["passes_fdr_q"] = q_value < config.TEMPORAL_FDR_SIGNIFICANCE_LEVEL
        if row["passes_negative_slope"] and row["passes_decay"] and row["passes_fdr_q"]:
            signals.append(row)
    report.update({
        "status": "completed",
        "recent_covered_years": recent_years,
        "historical_covered_years": historical_years,
        "pipeline_counts": {
            "non_generic_nodes": counts["non_generic_nodes"],
            "nodes_with_minimum_unique_papers": counts["nodes_with_minimum_unique_papers"],
            "eligible_nodes_tested": counts["eligible_nodes_tested"],
            "final_signals": len(signals),
        },
        "eligible_node_statistics": candidates,
        "signals": signals,
    })
    return report


def build_forward_sensitivity_report(resolved_triples: list[dict[str, Any]]) -> dict[str, Any]:
    graph = build_graph(resolved_triples)
    return {
        "schema_version": "forward-sensitivity-diagnostic-v1",
        "generated_at_utc": utc_now_iso(),
        "scope": "Exploratory robustness diagnostics; no candidate-quality or generalizability claim.",
        "structural_gate": structural_gate_diagnostic(graph),
        "paper_normalized_temporal": paper_normalized_temporal_diagnostic(
            graph, cutoff_year=config.TEMPORAL_CUTOFF_YEAR
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-generative forward sensitivity diagnostics.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(config.DATA_DIR) / "forward_sensitivity_diagnostic.json"
    )
    args = parser.parse_args()
    if not config.RUN_ID:
        parser.error("--run-id is required")
    triples_path = Path(config.TRIPLES_DIR) / "resolved_triples.json"
    triples = _load_json(triples_path)
    report = build_forward_sensitivity_report(triples)
    report["input_resolved_triples_sha256"] = file_sha256(triples_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Wrote forward sensitivity diagnostic to {args.output}")


if __name__ == "__main__":
    main()
