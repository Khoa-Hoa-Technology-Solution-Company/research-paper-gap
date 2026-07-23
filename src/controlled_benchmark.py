"""Offline structural-isolation perturbation benchmark for KG-TABI.

This benchmark evaluates only the topology-stage orphan-cluster detector.  It
does not evaluate LLM-generated TABI claims, novelty, factual support, or
research-gap validity.  The benchmark deliberately keeps the baseline Louvain
partition separate from the two measurements it reports:

* ``metric_only`` holds the baseline membership fixed and re-evaluates the
  size/bridge-ratio rule after masking.  It therefore tests the isolation
  metric itself.
* ``end_to_end`` reruns Louvain after masking and matches detected clusters to
  the pre-mask targets with one-to-one Jaccard matching.  It tests the complete
  topology stage, including partition instability.

The microservices-security-v1 run has only three communities that satisfy the
configured 5% size rule.  Rather than claim a 20--30-community sample that the
run cannot supply, this script evaluates every eligible community over 30
reproducible masking seeds.  It also writes both seeded trial records and
deduplicated-perturbation summaries so repeated 100% masks cannot create
spurious precision in confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def _configure_run_id_from_cli() -> None:
    """Read --run-id before importing config, whose paths are import-time values."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id", default=None)
    args, _ = parser.parse_known_args()
    if args.run_id:
        os.environ["KG_TABI_RUN_ID"] = args.run_id


_configure_run_id_from_cli()

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.graph_analysis import build_graph, louvain_partition


DEFAULT_MASK_LEVELS = (0.25, 0.50, 0.75, 1.00)
DEFAULT_JACCARD_THRESHOLD = 0.80
DEFAULT_MASKING_SEEDS = tuple(range(30))
DEFAULT_BOOTSTRAP_ITERATIONS = 2_000
WILSON_Z_95 = 1.959963984540054


def _node_key(node: Any) -> tuple[str, str]:
    return (str(node).lower(), str(node))


def _edge_key(source: Any, target: Any) -> tuple[Any, Any]:
    """Return a deterministic key for an undirected adjacency."""
    return (source, target) if _node_key(source) <= _node_key(target) else (target, source)


def _stable_seed(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _partition_to_communities(partition: dict[Any, int]) -> dict[int, frozenset[Any]]:
    grouped: dict[int, set[Any]] = defaultdict(set)
    for node, community_id in partition.items():
        grouped[community_id].add(node)
    return {community_id: frozenset(nodes) for community_id, nodes in grouped.items()}


def _undirected_edge_groups(graph) -> dict[tuple[Any, Any], tuple[tuple[Any, Any], ...]]:
    """Group all directed arcs that share one undirected adjacency.

    Louvain and the bridge ratio operate on ``G.to_undirected()``.  Masking a
    single directed arc would leave a reciprocal arc visible to that detector;
    a sampled unit must consequently remove every directed arc for an
    undirected adjacency.
    """
    grouped: dict[tuple[Any, Any], list[tuple[Any, Any]]] = defaultdict(list)
    for source, target in graph.edges():
        grouped[_edge_key(source, target)].append((source, target))
    return {
        key: tuple(sorted(arcs, key=lambda arc: (_node_key(arc[0]), _node_key(arc[1]))))
        for key, arcs in grouped.items()
    }


def _bridge_statistics(undirected_graph, nodes: frozenset[Any]) -> tuple[int, int, float]:
    internal_edges = 0
    cross_edges = 0
    for source, target in undirected_graph.edges():
        source_inside = source in nodes
        target_inside = target in nodes
        if source_inside and target_inside:
            internal_edges += 1
        elif source_inside != target_inside:
            cross_edges += 1
    denominator = internal_edges + cross_edges
    ratio = cross_edges / denominator if denominator else 0.0
    return internal_edges, cross_edges, ratio


def _detect_orphans_quiet(graph, partition: dict[Any, int]) -> list[dict[str, Any]]:
    """Apply the production orphan rule without the production progress logs."""
    total_nodes = graph.number_of_nodes()
    undirected = graph.to_undirected()
    predictions: list[dict[str, Any]] = []
    for community_id, nodes in _partition_to_communities(partition).items():
        size_ratio = len(nodes) / total_nodes
        internal, cross, bridge = _bridge_statistics(undirected, nodes)
        if (
            size_ratio >= config.LOUVAIN_MIN_SIZE_RATIO
            and bridge <= config.LOUVAIN_MAX_BRIDGE_RATIO
        ):
            predictions.append(
                {
                    "id": int(community_id),
                    "nodes": nodes,
                    "size": len(nodes),
                    "size_ratio": size_ratio,
                    "internal_undirected_edges": internal,
                    "cross_undirected_edges": cross,
                    "bridge_ratio": bridge,
                }
            )
    return sorted(predictions, key=lambda row: (-row["size"], row["id"]))


def _eligible_baseline_communities(
    graph, partition: dict[Any, int]
) -> list[dict[str, Any]]:
    total_nodes = graph.number_of_nodes()
    undirected = graph.to_undirected()
    rows: list[dict[str, Any]] = []
    for community_id, nodes in _partition_to_communities(partition).items():
        size_ratio = len(nodes) / total_nodes
        internal, cross, bridge = _bridge_statistics(undirected, nodes)
        if size_ratio >= config.LOUVAIN_MIN_SIZE_RATIO:
            rows.append(
                {
                    "id": int(community_id),
                    "nodes": nodes,
                    "size": len(nodes),
                    "size_ratio": size_ratio,
                    "internal_undirected_edges": internal,
                    "cross_undirected_edges": cross,
                    "bridge_ratio": bridge,
                }
            )
    return sorted(rows, key=lambda row: (-row["size"], row["id"]))


def _jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _one_to_one_matches(
    predictions: list[dict[str, Any]],
    baseline_communities: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Greedily take descending-Jaccard pairs, with deterministic tie breaks.

    A prediction and a baseline target can each participate in at most one
    match.  This prevents a fragmented target or one broad detected cluster
    from inflating the true-positive count.
    """
    candidates: list[tuple[float, int, int, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for baseline_index, baseline in enumerate(baseline_communities):
            overlap = len(prediction["nodes"] & baseline["nodes"])
            score = _jaccard(prediction["nodes"], baseline["nodes"])
            if score >= threshold:
                candidates.append(
                    (
                        score,
                        overlap,
                        prediction_index,
                        baseline_index,
                        prediction["size"],
                    )
                )
    candidates.sort(
        key=lambda row: (
            -row[0],
            -row[1],
            -row[4],
            str(predictions[row[2]]["id"]),
            str(baseline_communities[row[3]]["id"]),
        )
    )

    used_predictions: set[int] = set()
    used_baselines: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, overlap, prediction_index, baseline_index, _ in candidates:
        if prediction_index in used_predictions or baseline_index in used_baselines:
            continue
        used_predictions.add(prediction_index)
        used_baselines.add(baseline_index)
        matches.append(
            {
                "predicted_cluster_id": predictions[prediction_index]["id"],
                "baseline_community_id": baseline_communities[baseline_index]["id"],
                "jaccard": round(score, 8),
                "intersection_nodes": overlap,
            }
        )
    return matches


def _evaluate_predictions(
    predictions: list[dict[str, Any]],
    eligible_baselines: list[dict[str, Any]],
    induced_positive_ids: set[int],
    jaccard_threshold: float,
) -> dict[str, Any]:
    """Score outputs using induced targets only for precision and recall.

    Every output other than a one-to-one Jaccard match to an induced target is
    a false positive for precision, including an otherwise plausible orphan
    cluster.  For diagnostic purposes we separately count matches to other
    baseline-eligible communities and unmatched outputs.
    """
    matches = _one_to_one_matches(predictions, eligible_baselines, jaccard_threshold)
    matched_positive_ids = {
        match["baseline_community_id"]
        for match in matches
        if match["baseline_community_id"] in induced_positive_ids
    }
    true_positives = len(matched_positive_ids)
    false_negatives = len(induced_positive_ids) - true_positives
    false_positives = len(predictions) - true_positives
    matched_non_target_baselines = sum(
        match["baseline_community_id"] not in induced_positive_ids for match in matches
    )
    unmatched_outputs = len(predictions) - len(matches)
    baseline_negative_opportunities = len(eligible_baselines) - len(induced_positive_ids)

    best_target_jaccard: float | None = None
    if induced_positive_ids:
        target_nodes = [
            row["nodes"] for row in eligible_baselines if row["id"] in induced_positive_ids
        ]
        best_target_jaccard = max(
            (_jaccard(prediction["nodes"], target)
             for prediction in predictions for target in target_nodes),
            default=0.0,
        )

    return {
        "predicted_cluster_count": len(predictions),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matched_non_target_baselines": matched_non_target_baselines,
        "baseline_negative_opportunities": baseline_negative_opportunities,
        "unmatched_outputs": unmatched_outputs,
        "best_target_jaccard": (
            round(best_target_jaccard, 8) if best_target_jaccard is not None else None
        ),
        "matches": matches,
        "predicted_clusters": [
            {
                "id": row["id"],
                "size": row["size"],
                "bridge_ratio": round(row["bridge_ratio"], 8),
            }
            for row in predictions
        ],
    }


def _mask_graph(
    graph,
    edge_groups: dict[tuple[Any, Any], tuple[tuple[Any, Any], ...]],
    selected_edges: Iterable[tuple[Any, Any]],
) -> tuple[Any, int]:
    masked = graph.copy()
    removed_directed = 0
    for undirected_edge in selected_edges:
        for source, target in edge_groups[undirected_edge]:
            if masked.has_edge(source, target):
                masked.remove_edge(source, target)
                removed_directed += 1
    return masked, removed_directed


def _edge_signature(selected_edges: Iterable[tuple[Any, Any]]) -> str:
    material = "\n".join(
        f"{source}\u241f{target}"
        for source, target in sorted(selected_edges, key=lambda edge: (_node_key(edge[0]), _node_key(edge[1])))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _sample_edges(
    pool: list[tuple[Any, Any]],
    count: int,
    *seed_parts: Any,
) -> list[tuple[Any, Any]]:
    if count > len(pool):
        raise ValueError(f"Asked to sample {count} edges from a pool of {len(pool)}.")
    ordered_pool = sorted(pool, key=lambda edge: (_node_key(edge[0]), _node_key(edge[1])))
    rng = random.Random(_stable_seed(*seed_parts))
    return sorted(rng.sample(ordered_pool, count), key=lambda edge: (_node_key(edge[0]), _node_key(edge[1])))


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    if trials <= 0:
        return None
    proportion = successes / trials
    denominator = 1.0 + WILSON_Z_95**2 / trials
    centre = (proportion + WILSON_Z_95**2 / (2 * trials)) / denominator
    half_width = (
        WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + WILSON_Z_95**2 / (4 * trials**2)
        )
        / denominator
    )
    return [round(max(0.0, centre - half_width), 8), round(min(1.0, centre + half_width), 8)]


def _point_metrics(entries: list[dict[str, Any]]) -> dict[str, float | int | None]:
    true_positives = sum(entry["true_positives"] for entry in entries)
    false_positives = sum(entry["false_positives"] for entry in entries)
    false_negatives = sum(entry["false_negatives"] for entry in entries)
    predicted = sum(entry["predicted_cluster_count"] for entry in entries)
    precision = true_positives / predicted if predicted else None
    recall_denominator = true_positives + false_negatives
    recall = true_positives / recall_denominator if recall_denominator else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_clusters": predicted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _bootstrap_intervals(
    entries: list[dict[str, Any]], iterations: int, *seed_parts: Any
) -> dict[str, list[float] | None]:
    """Percentile bootstrap intervals over distinct perturbations only."""
    if not entries or iterations <= 0:
        return {"precision": None, "recall": None, "f1": None}
    rng = random.Random(_stable_seed("bootstrap", *seed_parts))
    values: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(iterations):
        sample = [entries[rng.randrange(len(entries))] for _ in range(len(entries))]
        metrics = _point_metrics(sample)
        for name in values:
            value = metrics[name]
            if value is not None:
                values[name].append(float(value))

    intervals: dict[str, list[float] | None] = {}
    for name, samples in values.items():
        if not samples:
            intervals[name] = None
            continue
        samples.sort()
        lower = samples[int(0.025 * (len(samples) - 1))]
        upper = samples[int(0.975 * (len(samples) - 1))]
        intervals[name] = [round(lower, 8), round(upper, 8)]
    return intervals


def _deduplicate_perturbations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one deterministic record per distinct selected edge set and target."""
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(record["perturbation_key"], record)
    return list(unique.values())


def _summarize_entries(
    records: list[dict[str, Any]], mode: str, bootstrap_iterations: int, *seed_parts: Any
) -> dict[str, Any]:
    entries = [record[mode] for record in records]
    points = _point_metrics(entries)
    any_output_successes = sum(entry["predicted_cluster_count"] > 0 for entry in entries)
    baseline_negative_matches = sum(entry["matched_non_target_baselines"] for entry in entries)
    baseline_negative_opportunities = sum(
        entry["baseline_negative_opportunities"] for entry in entries
    )
    best_jaccards = [
        entry["best_target_jaccard"]
        for entry in entries
        if entry["best_target_jaccard"] is not None
    ]
    points.update(
        {
            "mean_predicted_clusters_per_perturbation": (
                sum(entry["predicted_cluster_count"] for entry in entries) / len(entries)
                if entries else None
            ),
            "any_output_rate": any_output_successes / len(entries) if entries else None,
            "any_output_wilson_95_ci": _wilson_interval(any_output_successes, len(entries)),
            "matched_baseline_negative_communities": baseline_negative_matches,
            "baseline_negative_community_opportunities": baseline_negative_opportunities,
            "matched_baseline_unit_false_positive_rate": (
                baseline_negative_matches / baseline_negative_opportunities
                if baseline_negative_opportunities else None
            ),
            "matched_baseline_unit_fpr_wilson_95_ci": _wilson_interval(
                baseline_negative_matches, baseline_negative_opportunities
            ),
            "unmatched_orphan_outputs": sum(entry["unmatched_outputs"] for entry in entries),
            "mean_best_target_jaccard": mean(best_jaccards) if best_jaccards else None,
            "median_best_target_jaccard": median(best_jaccards) if best_jaccards else None,
            "metric_bootstrap_95_ci": _bootstrap_intervals(
                entries, bootstrap_iterations, mode, *seed_parts
            ),
        }
    )
    for name, value in list(points.items()):
        if isinstance(value, float):
            points[name] = round(value, 8)
    return points


def _group_summaries(
    trial_records: list[dict[str, Any]], mode: str, bootstrap_iterations: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in trial_records:
        grouped[(record["condition"], record["mask_level"])].append(record)

    summaries: list[dict[str, Any]] = []
    for (condition, level), records in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        unique_records = _deduplicate_perturbations(records)
        summary = {
            "condition": condition,
            "mask_level": level,
            "nominal_seeded_trials": len(records),
            "unique_perturbations": len(unique_records),
            "distinct_targets": len({record["target"]["baseline_community_id"] for record in records}),
            "primary_unique_perturbation_metrics": _summarize_entries(
                unique_records, mode, bootstrap_iterations, condition, level, "unique"
            ),
            "seeded_trial_diagnostic_metrics": _summarize_entries(
                records, mode, 0, condition, level, "seeded"
            ),
        }
        summaries.append(summary)
    return summaries


def _json_ready_baseline(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "community_id": row["id"],
        "size": row["size"],
        "size_ratio": round(row["size_ratio"], 8),
        "internal_undirected_edges": row["internal_undirected_edges"],
        "cross_undirected_edges": row["cross_undirected_edges"],
        "bridge_ratio": round(row["bridge_ratio"], 8),
    }


def _write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_controlled_benchmark(
    masking_seeds: Iterable[int] = DEFAULT_MASKING_SEEDS,
    masking_levels: Iterable[float] = DEFAULT_MASK_LEVELS,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Run the multi-condition offline structural perturbation benchmark."""
    seeds = tuple(int(seed) for seed in masking_seeds)
    levels = tuple(sorted({float(level) for level in masking_levels}))
    if not seeds:
        raise ValueError("At least one masking seed is required.")
    if not levels or any(level <= 0.0 or level > 1.0 for level in levels):
        raise ValueError("Masking levels must lie in (0, 1].")
    if not 0.0 < jaccard_threshold <= 1.0:
        raise ValueError("Jaccard threshold must lie in (0, 1].")

    resolved_path = os.path.join(config.TRIPLES_DIR, "resolved_triples.json")
    with open(resolved_path, encoding="utf-8") as handle:
        graph = build_graph(json.load(handle))
    baseline_partition = louvain_partition(graph)
    edge_groups = _undirected_edge_groups(graph)
    all_undirected_edges = sorted(edge_groups, key=lambda edge: (_node_key(edge[0]), _node_key(edge[1])))
    eligible = _eligible_baseline_communities(graph, baseline_partition)
    baseline_orphans = _detect_orphans_quiet(graph, baseline_partition)

    minimum_size_nodes = math.ceil(config.LOUVAIN_MIN_SIZE_RATIO * graph.number_of_nodes())
    target_catalog: list[dict[str, Any]] = []
    excluded_eligible_communities: list[dict[str, Any]] = []
    for community in eligible:
        nodes = community["nodes"]
        boundary_pool = [
            edge for edge in all_undirected_edges if (edge[0] in nodes) != (edge[1] in nodes)
        ]
        internal_pool = [
            edge for edge in all_undirected_edges if edge[0] in nodes and edge[1] in nodes
        ]
        max_requested = math.ceil(max(levels) * len(boundary_pool)) if boundary_pool else 0
        if not boundary_pool:
            excluded_eligible_communities.append(
                {
                    **_json_ready_baseline(community),
                    "reason": "no boundary adjacency to perturb",
                }
            )
            continue
        if len(internal_pool) < max_requested:
            excluded_eligible_communities.append(
                {
                    **_json_ready_baseline(community),
                    "reason": "insufficient internal adjacencies for same-count internal-edge control",
                    "available_internal_undirected_edges": len(internal_pool),
                    "maximum_requested_removal": max_requested,
                }
            )
            continue
        target_catalog.append(
            {
                **community,
                "boundary_pool": boundary_pool,
                "internal_pool": internal_pool,
            }
        )

    if not target_catalog:
        raise RuntimeError("No eligible baseline community supports all requested controls.")

    conditions = {
        "targeted_boundary": {
            "description": "Randomly removes selected target-community boundary adjacencies; the selected target is the sole induced positive.",
            "positive_target": True,
        },
        "internal_edge_control": {
            "description": "Removes the same number of target-internal adjacencies while retaining all target boundary adjacencies; no induced positives.",
            "positive_target": False,
        },
        "global_random_edge_control": {
            "description": "Removes the same number of uniformly sampled graph adjacencies without targeting a community boundary; no induced positives.",
            "positive_target": False,
        },
    }

    trial_records: list[dict[str, Any]] = []
    planned_trials = len(conditions) * len(levels) * len(seeds) * len(target_catalog)
    completed_trials = 0
    for condition, condition_spec in conditions.items():
        for level in levels:
            for target in target_catalog:
                removal_count = math.ceil(level * len(target["boundary_pool"]))
                for masking_seed in seeds:
                    if condition == "targeted_boundary":
                        selected_edges = _sample_edges(
                            target["boundary_pool"],
                            removal_count,
                            "boundary",
                            target["id"],
                            level,
                            masking_seed,
                        )
                    elif condition == "internal_edge_control":
                        selected_edges = _sample_edges(
                            target["internal_pool"],
                            removal_count,
                            "internal",
                            target["id"],
                            level,
                            masking_seed,
                        )
                    else:
                        selected_edges = _sample_edges(
                            all_undirected_edges,
                            removal_count,
                            "global",
                            target["id"],
                            level,
                            masking_seed,
                        )

                    masked_graph, removed_directed = _mask_graph(graph, edge_groups, selected_edges)
                    induced_positive_ids = {target["id"]} if condition_spec["positive_target"] else set()
                    metric_only_predictions = _detect_orphans_quiet(masked_graph, baseline_partition)
                    end_to_end_predictions = _detect_orphans_quiet(
                        masked_graph, louvain_partition(masked_graph)
                    )
                    signature = _edge_signature(selected_edges)
                    target_json = _json_ready_baseline(target)
                    target_json.update(
                        {
                            "baseline_community_id": target["id"],
                            "boundary_pool_undirected_edges": len(target["boundary_pool"]),
                            "internal_pool_undirected_edges": len(target["internal_pool"]),
                        }
                    )
                    trial_records.append(
                        {
                            "condition": condition,
                            "mask_level": level,
                            "masking_seed": masking_seed,
                            "target": target_json,
                            "induced_positive_baseline_community_ids": sorted(induced_positive_ids),
                            "removed_undirected_edges": len(selected_edges),
                            "removed_directed_edges": removed_directed,
                            "actual_removed_boundary_fraction": round(
                                len(selected_edges) / len(target["boundary_pool"]), 8
                            ),
                            "mask_signature_sha256": signature,
                            "perturbation_key": f"{condition}|{level}|{target['id']}|{signature}",
                            "metric_only": _evaluate_predictions(
                                metric_only_predictions,
                                eligible,
                                induced_positive_ids,
                                jaccard_threshold,
                            ),
                            "end_to_end": _evaluate_predictions(
                                end_to_end_predictions,
                                eligible,
                                induced_positive_ids,
                                jaccard_threshold,
                            ),
                        }
                    )
                    completed_trials += 1
                    if completed_trials % 120 == 0 or completed_trials == planned_trials:
                        print(f"[*] Completed {completed_trials}/{planned_trials} perturbation trials.")

    result_path = os.path.join(config.DATA_DIR, "controlled_benchmark_multicondition_results.json")
    trials_path = os.path.join(config.DATA_DIR, "controlled_benchmark_multicondition_trials.jsonl")
    result: dict[str, Any] = {
        "benchmark": "multi-condition structural-isolation perturbation benchmark",
        "benchmark_version": "2.0",
        "scope": "Topology-stage structural-orphan detection only. It does not validate TABI claim generation, evidence support, novelty, or research-gap usefulness.",
        "run": {
            "run_id": config.RUN_ID or None,
            "data_directory": config.DATA_DIR,
            "resolved_triples_path": resolved_path,
            "resolved_triples_sha256": _sha256_file(resolved_path),
            "nodes": graph.number_of_nodes(),
            "directed_edges": graph.number_of_edges(),
            "undirected_adjacencies": len(edge_groups),
        },
        "design": {
            "baseline_partition_source": "Louvain on the unmasked graph using the configured fixed random state",
            "louvain_random_state": config.LOUVAIN_RANDOM_STATE,
            "louvain_resolution": config.LOUVAIN_RESOLUTION,
            "orphan_rule": {
                "minimum_size_ratio": config.LOUVAIN_MIN_SIZE_RATIO,
                "minimum_size_nodes_for_this_graph": minimum_size_nodes,
                "maximum_symmetric_bridge_ratio": config.LOUVAIN_MAX_BRIDGE_RATIO,
                "projection": "undirected",
            },
            "target_selection": {
                "policy": "all baseline Louvain communities meeting the configured size rule and supporting every same-count control",
                "eligible_baseline_communities": len(eligible),
                "tested_target_communities": len(target_catalog),
                "excluded_eligible_communities": excluded_eligible_communities,
                "why_not_20_to_30_targets": "The configured 5% size rule yields fewer than 20 eligible communities in this run; all eligible, control-feasible communities were used instead.",
            },
            "masking": {
                "levels": list(levels),
                "seeds": list(seeds),
                "seed_count": len(seeds),
                "unit": "undirected adjacency; all reciprocal directed arcs for a sampled adjacency are removed together",
                "rounding": "ceil(level * target boundary-adjacency count); actual fraction is recorded per trial",
            },
            "measurements": {
                "metric_only": "Holds the baseline Louvain membership fixed and reapplies the size/bridge-ratio detector after masking.",
                "end_to_end": "Reruns Louvain after masking, then matches output clusters to baseline communities.",
            },
            "matching": {
                "metric": "Jaccard node-set similarity",
                "threshold": jaccard_threshold,
                "rule": "Descending-Jaccard greedy one-to-one matching; ties break by overlap, detected size, detected ID, then baseline ID.",
                "precision_labeling": "Only deliberately boundary-masked target communities are positives. Every other detected orphan output is a false positive, including a match to a non-target baseline community.",
            },
            "negative_controls": [
                conditions["internal_edge_control"]["description"],
                conditions["global_random_edge_control"]["description"],
            ],
            "intervals": {
                "metric_intervals": "2,000-resample percentile bootstrap over unique perturbations, not repeated identical seed trials.",
                "rate_intervals": "Wilson 95% intervals; descriptive because target communities derive from this single graph.",
                "bootstrap_iterations": bootstrap_iterations,
            },
        },
        "baseline": {
            "community_count": len(_partition_to_communities(baseline_partition)),
            "eligible_communities": [_json_ready_baseline(row) for row in eligible],
            "baseline_orphan_count": len(baseline_orphans),
            "baseline_orphan_community_ids": [row["id"] for row in baseline_orphans],
        },
        "summary": {
            "metric_only": _group_summaries(trial_records, "metric_only", bootstrap_iterations),
            "end_to_end": _group_summaries(trial_records, "end_to_end", bootstrap_iterations),
        },
        "limitations": [
            "Targets originate from the baseline Louvain partition, so this is a conditional perturbation robustness test rather than an external ground-truth benchmark.",
            "Only three communities satisfy the configured 5% size rule in this run; results cannot establish population-level generalization.",
            "Masked edges are observed graph adjacencies, not verified missing scientific relations.",
            "The negative controls test structural specificity under graph perturbation; they do not validate downstream TABI text generation.",
            "Repeated seeds can yield the same finite edge mask. Primary summaries and bootstrap intervals deduplicate identical perturbations to avoid treating repetitions as independent samples.",
        ],
        "artifacts": {
            "summary_json": os.path.basename(result_path),
            "trial_records_jsonl": os.path.basename(trials_path),
            "nominal_seeded_trials": len(trial_records),
            "primary_analysis_unit": "unique target-condition-level-selected-edge-set perturbation",
        },
    }
    _write_jsonl(trials_path, trial_records)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"[+] Wrote benchmark summary to {result_path}")
    print(f"[+] Wrote seeded trial records to {trials_path}")
    return result


def _parse_levels(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Mask levels must be comma-separated floats.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=config.RUN_ID or None,
        help="Artifact run ID (must also be visible before config import; this module handles it automatically).",
    )
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_MASKING_SEEDS))
    parser.add_argument(
        "--mask-levels",
        type=_parse_levels,
        default=DEFAULT_MASK_LEVELS,
        help="Comma-separated fractions, e.g. 0.25,0.50,0.75,1.00.",
    )
    parser.add_argument("--jaccard-threshold", type=float, default=DEFAULT_JACCARD_THRESHOLD)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.bootstrap_iterations < 0:
        parser.error("--bootstrap-iterations cannot be negative")
    run_controlled_benchmark(
        masking_seeds=range(args.seeds),
        masking_levels=args.mask_levels,
        jaccard_threshold=args.jaccard_threshold,
        bootstrap_iterations=args.bootstrap_iterations,
    )


if __name__ == "__main__":
    main()
