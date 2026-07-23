"""Traceability-restricted sensitivity for the legacy pilot graph.

The legacy pilot contains 800 resolved triples, but only 774 can be linked
post hoc to exactly one retained abstract chunk by a case- and
whitespace-normalised exact evidence-quote match.  This module never changes
the pilot artifacts.  It rebuilds two graphs side by side:

* ``full_legacy``: all 800 legacy resolved triples;
* ``traceable_only``: only triples with that unique recovered route.

It then applies the configured deterministic topology and temporal stages to
both graphs.  The resulting report is a sensitivity analysis of *source-route
availability*, not an evidence-support audit: a quote occurring in a chunk
does not establish that the quote supports the extracted triple.

An optional, explicitly exploratory degree-preserving null diagnostic is also
provided.  It rewires the simple undirected projection, reruns Louvain for
each null graph, and compares the minimum cut-edge fraction among
size-eligible communities.  It preserves only the degree sequence, so it is
not a calibration or validation of the orphan threshold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

import networkx as nx

from src import config
from src.graph_analysis import (
    build_graph,
    community_cut_edge_metrics,
    compute_temporal_decay,
    louvain_partition,
    topology_configuration_hash,
)
from src.robustness_analysis import (
    _indices_align,
    adjusted_rand_index,
    recover_quote_provenance,
)


METHOD_VERSION = "legacy-traceability-sensitivity-v1"
UNIQUE_ROUTE_STATUS = "unique_normalized_exact_quote_match"
DEFAULT_NULL_SAMPLES = 500
DEFAULT_SWAPS_PER_EDGE = 10
DEFAULT_NULL_SEED = 20260720


def _load_json(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_to_run(path: str) -> str:
    return os.path.relpath(path, config.DATA_DIR).replace("\\", "/")


def _load_traceable_indices(
    audit_path: str,
    raw_triples: list[dict[str, Any]],
    resolved_triples: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[set[int], dict[str, Any]]:
    """Read and cross-validate the recovery CSV before filtering triples.

    The CSV is treated as an audit artifact, not as an opaque filtering list.
    Every row must correspond to one legacy triple, must agree with its raw
    fields, and must agree with a fresh conservative recovery calculation.
    This prevents accidental index drift between raw and resolved triples.
    """
    if not _indices_align(raw_triples, resolved_triples):
        raise RuntimeError(
            "Raw and resolved triples are not index-aligned; refusing to transfer "
            "post-hoc provenance to the resolved graph."
        )

    required_columns = {
        "legacy_triple_index",
        "subject",
        "relation",
        "object",
        "publication_year",
        "evidence_quote",
        "provenance_recovery_status",
        "paper_id",
        "chunk_index",
    }
    with open(audit_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            missing = sorted(required_columns - set(reader.fieldnames or ()))
            raise RuntimeError(f"Provenance recovery CSV lacks required columns: {missing}")
        rows = list(reader)

    by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            index = int(row["legacy_triple_index"])
        except (TypeError, ValueError) as error:
            raise RuntimeError("A recovery CSV index is not an integer.") from error
        if index in by_index or index < 0 or index >= len(raw_triples):
            raise RuntimeError("Recovery CSV has duplicate or out-of-range triple indexes.")
        by_index[index] = row

    expected_indexes = set(range(len(raw_triples)))
    if set(by_index) != expected_indexes:
        missing = len(expected_indexes - set(by_index))
        extra = len(set(by_index) - expected_indexes)
        raise RuntimeError(
            f"Recovery CSV must cover every legacy triple exactly once (missing={missing}, extra={extra})."
        )

    recomputed = recover_quote_provenance(raw_triples, chunks)
    audit_statuses = Counter()
    for index, raw in enumerate(raw_triples):
        row = by_index[index]
        for csv_key, raw_key in (
            ("subject", "subject"),
            ("relation", "relation"),
            ("object", "object"),
            ("evidence_quote", "evidence_quote"),
        ):
            if row[csv_key] != str(raw.get(raw_key, "")):
                raise RuntimeError(
                    f"Recovery CSV {csv_key} disagrees with raw triple at index {index}."
                )
        if row["publication_year"] != str(raw.get("year", "")):
            raise RuntimeError(
                f"Recovery CSV publication year disagrees with raw triple at index {index}."
            )

        recovered = recomputed[index]
        if row["provenance_recovery_status"] != recovered.status:
            raise RuntimeError(
                f"Recovery status disagrees with a fresh exact-quote lookup at index {index}."
            )
        if recovered.status == UNIQUE_ROUTE_STATUS:
            if row["paper_id"] != str(recovered.paper_id or ""):
                raise RuntimeError(
                    f"Recovered paper ID disagrees with a fresh lookup at index {index}."
                )
            if row["chunk_index"] != str(recovered.chunk_index):
                raise RuntimeError(
                    f"Recovered chunk index disagrees with a fresh lookup at index {index}."
                )
        audit_statuses[row["provenance_recovery_status"]] += 1

    traceable = {
        index
        for index, row in by_index.items()
        if row["provenance_recovery_status"] == UNIQUE_ROUTE_STATUS
    }
    return traceable, {
        "raw_resolved_index_alignment": True,
        "recovery_csv_covers_all_legacy_triples": True,
        "fresh_exact_quote_lookup_agrees_with_recovery_csv": True,
        "recovery_method": (
            "unique case/whitespace-normalized exact evidence-quote substring "
            "match in retained chunks"
        ),
        "recovery_status_counts": dict(sorted(audit_statuses.items())),
        "traceable_triple_count": len(traceable),
        "untraceable_triple_count": len(raw_triples) - len(traceable),
    }


def _communities(partition: dict[Any, int]) -> dict[int, set[Any]]:
    result: dict[int, set[Any]] = defaultdict(set)
    for node, community_id in partition.items():
        result[community_id].add(node)
    return dict(result)


def _cut_edge_metrics(
    graph: nx.Graph,
    partition: dict[Any, int],
) -> list[dict[str, Any]]:
    """Compatibility wrapper around the shared production statistic."""
    return community_cut_edge_metrics(graph, partition)


def _summarize_graph(
    label: str,
    triples: list[dict[str, Any]],
) -> tuple[dict[str, Any], nx.DiGraph, dict[Any, int], list[dict[str, Any]]]:
    """Run the production topology and temporal analyses on one triple set."""
    graph = build_graph(triples)
    projection = graph.to_undirected()
    partition = louvain_partition(graph)
    communities = _communities(partition)
    cut_metrics = _cut_edge_metrics(projection, partition)
    eligible_by_size = [
        item for item in cut_metrics
        if item["size_ratio"] >= config.LOUVAIN_MIN_SIZE_RATIO
    ]
    orphans = [
        item for item in eligible_by_size
        if item["cut_edge_fraction"] <= config.LOUVAIN_MAX_BRIDGE_RATIO
    ]
    temporal_signals, temporal_report = compute_temporal_decay(graph, return_report=True)
    temporal_counts = temporal_report["pipeline_counts"]
    modularity = nx.algorithms.community.quality.modularity(
        projection,
        list(communities.values()),
        weight=None,
    )
    summary = {
        "label": label,
        "input_triples": len(triples),
        "nodes": graph.number_of_nodes(),
        "directed_edges": graph.number_of_edges(),
        "undirected_projection_edges": projection.number_of_edges(),
        "undirected_projection_connected_components": nx.number_connected_components(projection),
        "louvain": {
            "implementation": "python-louvain.best_partition",
            "random_state": config.LOUVAIN_RANDOM_STATE,
            "resolution": config.LOUVAIN_RESOLUTION,
            "community_count": len(communities),
            "modularity": round(modularity, 8),
            "size_eligible_community_count": len(eligible_by_size),
            "minimum_size_eligible_cut_edge_fraction": (
                round(min(item["cut_edge_fraction"] for item in eligible_by_size), 8)
                if eligible_by_size else None
            ),
            "orphan_count": len(orphans),
            "orphan_rule": {
                "minimum_size_ratio": config.LOUVAIN_MIN_SIZE_RATIO,
                "maximum_cut_edge_fraction": config.LOUVAIN_MAX_BRIDGE_RATIO,
            },
        },
        "temporal": {
            "status": temporal_report["status"],
            "non_generic_nodes": temporal_counts["non_generic_nodes"],
            "nodes_with_minimum_events": temporal_counts["nodes_with_minimum_events"],
            "nodes_with_minimum_distinct_years": temporal_counts[
                "nodes_with_minimum_distinct_years"
            ],
            "eligible_nodes_tested": temporal_counts["eligible_nodes_tested"],
            "final_signal_count": len(temporal_signals),
            "pipeline_counts": temporal_counts,
        },
    }
    return summary, graph, partition, cut_metrics


def _summary(values: Iterable[float]) -> dict[str, float | None]:
    numbers = list(values)
    if not numbers:
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": round(min(numbers), 8),
        "median": round(statistics.median(numbers), 8),
        "maximum": round(max(numbers), 8),
    }


def _minimum_eligible_cut_fraction(
    graph: nx.Graph,
    partition: dict[Any, int],
) -> tuple[float | None, int, int]:
    metrics = _cut_edge_metrics(graph, partition)
    eligible = [
        item for item in metrics
        if item["size_ratio"] >= config.LOUVAIN_MIN_SIZE_RATIO
    ]
    if not eligible:
        return None, 0, 0
    return (
        min(float(item["cut_edge_fraction"]) for item in eligible),
        len(eligible),
        sum(
            item["cut_edge_fraction"] <= config.LOUVAIN_MAX_BRIDGE_RATIO
            for item in eligible
        ),
    )


def _degree_preserving_null_diagnostic(
    graph: nx.DiGraph,
    observed_partition: dict[Any, int],
    *,
    samples: int,
    swaps_per_edge: int,
    seed: int,
) -> dict[str, Any]:
    """Run a labelled exploratory rewiring diagnostic on an undirected projection.

    Each null starts from the observed simple projection and is independently
    rewired with ``double_edge_swap``.  Louvain is rerun after rewiring, so
    the global-minimum statistic includes the community-detection selection
    step rather than conditioning on one post-hoc observed membership.
    """
    if samples < 1:
        return {
            "status": "not_run",
            "reason": "null sample count is zero",
            "exploratory": True,
        }
    projection = graph.to_undirected()
    if nx.number_of_selfloops(projection):
        return {
            "status": "not_run",
            "reason": "the degree-preserving simple-graph swap is not defined here with self-loops",
            "exploratory": True,
        }
    observed_minimum, observed_eligible, observed_orphans = _minimum_eligible_cut_fraction(
        projection, observed_partition
    )
    if observed_minimum is None:
        return {
            "status": "not_run",
            "reason": "no observed communities pass the configured size rule",
            "exploratory": True,
        }

    base_degrees = sorted(degree for _, degree in projection.degree())
    swaps = swaps_per_edge * projection.number_of_edges()
    samples_out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample_index in range(samples):
        sample_seed = seed + sample_index
        null_graph = nx.Graph(projection)
        try:
            nx.double_edge_swap(
                null_graph,
                nswap=swaps,
                max_tries=max(swaps * 50, 1),
                seed=sample_seed,
            )
        except nx.NetworkXAlgorithmError as error:
            failures.append({"sample_index": sample_index, "seed": sample_seed, "error": str(error)})
            continue
        if (
            null_graph.number_of_nodes() != projection.number_of_nodes()
            or null_graph.number_of_edges() != projection.number_of_edges()
            or sorted(degree for _, degree in null_graph.degree()) != base_degrees
        ):
            raise RuntimeError("Degree-preserving null validation failed after edge rewiring.")
        null_partition = louvain_partition(null_graph)
        minimum, eligible_count, orphan_count = _minimum_eligible_cut_fraction(
            null_graph, null_partition
        )
        # The global low-tail statistic must be defined for every rewired
        # graph.  A null partition with no community satisfying the fixed
        # size rule cannot contain an eligible orphan, so it receives the
        # largest possible cut-edge fraction (1.0), rather than being dropped
        # from the Monte-Carlo reference distribution.
        no_size_eligible_community = minimum is None
        if minimum is None:
            minimum = 1.0
        samples_out.append({
            "sample_index": sample_index,
            "seed": sample_seed,
            "minimum_size_eligible_cut_edge_fraction": round(minimum, 8),
            "size_eligible_community_count": eligible_count,
            "orphan_count_at_configured_threshold": orphan_count,
            "no_size_eligible_community": no_size_eligible_community,
        })

    null_minima = [float(item["minimum_size_eligible_cut_edge_fraction"]) for item in samples_out]
    lower_tail_count = sum(value <= observed_minimum for value in null_minima)
    return {
        "status": "completed" if len(samples_out) == samples else "completed_with_dropped_samples",
        "exploratory": True,
        "statistic": (
            "minimum cut-edge fraction among communities satisfying the fixed "
            "minimum-size rule; Louvain is rerun for every null graph"
        ),
        "null_model": {
            "graph": "simple undirected projection of the directed knowledge graph",
            "rewiring": "networkx.double_edge_swap independently from the observed projection",
            "degree_sequence_preserved_for_every_completed_sample": True,
            "samples_requested": samples,
            "samples_completed": len(samples_out),
            "null_samples_without_a_size_eligible_community": sum(
                item["no_size_eligible_community"] for item in samples_out
            ),
            "swaps_per_edge": swaps_per_edge,
            "swaps_per_sample": swaps,
            "seed_start": seed,
        },
        "observed": {
            "minimum_size_eligible_cut_edge_fraction": round(observed_minimum, 8),
            "size_eligible_community_count": observed_eligible,
            "orphan_count_at_configured_threshold": observed_orphans,
        },
        "null_distribution": {
            "minimum_cut_edge_fraction": _summary(null_minima),
            "lower_or_equal_than_observed_count": lower_tail_count,
            "empirical_lower_tail_probability": round(
                (lower_tail_count + 1) / (len(null_minima) + 1), 8
            ) if null_minima else None,
            "orphan_count": _summary(
                float(item["orphan_count_at_configured_threshold"]) for item in samples_out
            ),
            "nonzero_orphan_rate": round(
                sum(item["orphan_count_at_configured_threshold"] > 0 for item in samples_out)
                / len(samples_out),
                8,
            ) if samples_out else None,
        },
        "not_reported": {
            "per_community_p_iso_or_fdr": (
                "Not reported. Observed communities are selected from the same graph by "
                "Louvain and do not have stable identities after rewiring, so matching them "
                "to null communities would create an additional post-hoc alignment rule. "
                "The single global-minimum statistic instead keeps the full selection step "
                "inside every null run."
            ),
        },
        "samples": samples_out,
        "dropped_samples": failures,
        "interpretation_guardrail": (
            "This is an exploratory conditional null diagnostic, not a calibrated or "
            "confirmatory p-value for a named community or the beta=0.10 threshold. "
            "It preserves only undirected degree sequence and reruns a selected Louvain "
            "partition; it does not preserve direction, relation/event multiplicity, "
            "years, entity types, semantic structure, or connected components. A null "
            "partition with no size-eligible community is assigned statistic 1.0 so it "
            "remains in this one-sided low-tail reference distribution."
        ),
    }


def run_traceability_sensitivity(
    *,
    null_samples: int = DEFAULT_NULL_SAMPLES,
    null_swaps_per_edge: int = DEFAULT_SWAPS_PER_EDGE,
    null_seed: int = DEFAULT_NULL_SEED,
    output: str | None = None,
) -> dict[str, Any]:
    """Build, compare, and persist the full and traceable-only legacy graphs."""
    raw_path = os.path.join(config.TRIPLES_DIR, "raw_triples.json")
    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    chunks_path = os.path.join(config.RAW_PAPERS_DIR, "chunks.json")
    audit_path = os.path.join(
        config.DATA_DIR, "audits", "legacy_triple_provenance_recovery.csv"
    )
    for path in (raw_path, resolved_path, chunks_path, audit_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required legacy sensitivity input does not exist: {path}")

    raw_triples = _load_json(raw_path)
    resolved_triples = _load_json(resolved_path)
    chunks = _load_json(chunks_path)
    traceable_indices, validation = _load_traceable_indices(
        audit_path, raw_triples, resolved_triples, chunks
    )
    traceable_triples = [
        triple for index, triple in enumerate(resolved_triples) if index in traceable_indices
    ]
    if len(traceable_triples) != validation["traceable_triple_count"]:
        raise RuntimeError("Traceable triple count changed while applying the validated filter.")

    print("[*] Building full legacy graph (800 triples) ...")
    full_summary, full_graph, full_partition, _ = _summarize_graph(
        "full_legacy", resolved_triples
    )
    print("[*] Building traceable-only graph (unique post-hoc routes only) ...")
    trace_summary, trace_graph, trace_partition, _ = _summarize_graph(
        "traceable_only", traceable_triples
    )

    full_nodes = set(full_graph.nodes())
    trace_nodes = set(trace_graph.nodes())
    shared_nodes = sorted(full_nodes & trace_nodes, key=lambda item: str(item).casefold())
    shared_full_partition = {node: full_partition[node] for node in shared_nodes}
    shared_trace_partition = {node: trace_partition[node] for node in shared_nodes}
    if shared_nodes:
        shared_ari = adjusted_rand_index(shared_full_partition, shared_trace_partition)
    else:  # pragma: no cover - impossible for this legacy artifact, defensive for future runs
        shared_ari = None

    print("[*] Running exploratory degree-preserving null diagnostics ...")
    null_diagnostics = {
        "full_legacy": _degree_preserving_null_diagnostic(
            full_graph,
            full_partition,
            samples=null_samples,
            swaps_per_edge=null_swaps_per_edge,
            seed=null_seed,
        ),
        "traceable_only": _degree_preserving_null_diagnostic(
            trace_graph,
            trace_partition,
            samples=null_samples,
            swaps_per_edge=null_swaps_per_edge,
            seed=null_seed + 100_000,
        ),
    }

    report = {
        "method_version": METHOD_VERSION,
        "scope": (
            "Offline traceability-restricted sensitivity for the legacy pilot. It compares "
            "topology and temporal outputs after removing triples without a unique post-hoc "
            "quote-to-chunk route. It does not establish triple correctness, evidence support, "
            "end-to-end reproducibility, or research-gap hypothesis quality."
        ),
        "input_artifacts": {
            _relative_to_run(raw_path): _sha256_file(raw_path),
            _relative_to_run(resolved_path): _sha256_file(resolved_path),
            _relative_to_run(chunks_path): _sha256_file(chunks_path),
            _relative_to_run(audit_path): _sha256_file(audit_path),
        },
        "provenance_filter_validation": validation,
        "configured_analysis": {
            "louvain_random_state": config.LOUVAIN_RANDOM_STATE,
            "louvain_resolution": config.LOUVAIN_RESOLUTION,
            "minimum_size_ratio": config.LOUVAIN_MIN_SIZE_RATIO,
            "maximum_cut_edge_fraction": config.LOUVAIN_MAX_BRIDGE_RATIO,
            "temporal_cutoff_year": config.TEMPORAL_CUTOFF_YEAR,
            "temporal_fdr_significance_level": config.TEMPORAL_FDR_SIGNIFICANCE_LEVEL,
            "topology_configuration_sha256": topology_configuration_hash(),
            "statistic_implementation": "src.graph_analysis.community_cut_edge_metrics",
        },
        "graph_comparison": {
            "full_legacy": full_summary,
            "traceable_only": trace_summary,
            "partition_comparison": {
                "comparison_basis": (
                    "Adjusted Rand index calculated only on nodes present in both graphs; "
                    "community numeric labels are ignored."
                ),
                "shared_node_count": len(shared_nodes),
                "full_only_node_count": len(full_nodes - trace_nodes),
                "traceable_only_node_count": len(trace_nodes - full_nodes),
                "adjusted_rand_index_on_shared_nodes": (
                    round(shared_ari, 8) if shared_ari is not None else None
                ),
            },
        },
        "exploratory_degree_preserving_null_diagnostic": null_diagnostics,
        "caveats": [
            "A unique exact quote-to-chunk route is a source-location recovery result, not a human judgment that the quote supports the triple.",
            "The legacy source model/version, prompt version, and source offsets remain unrecoverable; this analysis cannot make the pilot end-to-end reproducible.",
            "The traceable-only graph may still retain incorrect extractions or entity merges; removing the 26 unmatched triples is only one provenance sensitivity.",
            "Partition ARI excludes nodes absent from the traceable-only graph, so it measures membership stability on the shared node set only.",
            "The null diagnostic is exploratory and cannot calibrate beta=0.10 without a pre-specified development/held-out protocol and a null model justified for the target graph-generating process.",
        ],
    }
    destination = output or os.path.join(
        config.DATA_DIR, "traceability_restricted_sensitivity.json"
    )
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"[+] Saved traceability-restricted sensitivity report to {destination}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full legacy and traceable-only KG-TABI pilot graphs."
    )
    parser.add_argument(
        "--null-samples",
        type=int,
        default=DEFAULT_NULL_SAMPLES,
        help="Degree-preserving null samples per graph (default: 500). Use 0 to skip.",
    )
    parser.add_argument(
        "--null-swaps-per-edge",
        type=int,
        default=DEFAULT_SWAPS_PER_EDGE,
        help="Independent double-edge swaps per projection edge (default: 10).",
    )
    parser.add_argument(
        "--null-seed",
        type=int,
        default=DEFAULT_NULL_SEED,
        help="First deterministic seed for full-graph null samples.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON destination.")
    args = parser.parse_args()
    if args.null_samples < 0:
        parser.error("--null-samples must be non-negative")
    if args.null_swaps_per_edge < 1:
        parser.error("--null-swaps-per-edge must be at least 1")
    run_traceability_sensitivity(
        null_samples=args.null_samples,
        null_swaps_per_edge=args.null_swaps_per_edge,
        null_seed=args.null_seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
