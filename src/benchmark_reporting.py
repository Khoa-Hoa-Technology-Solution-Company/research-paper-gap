"""Create an appendix-ready report for the controlled perturbation benchmark.

The report is deliberately separate from :mod:`src.controlled_benchmark` so it
can be regenerated from the immutable seeded-trial and summary artifacts.  It
makes the primary analysis unit explicit, presents the existing bootstrap
intervals for both benchmark measurements, and adds only comparisons that are
well-defined without tuning a detector on the same perturbations.

In particular, the optional structural comparison is *metric-only*: it holds
the original Louvain partition fixed and ranks the three communities that pass
the pre-specified 5% size gate.  It does not purport to tune or validate a
conductance threshold.  ``external-edge fraction`` is reported as an identity
with the production bridge ratio, rather than as a misleadingly independent
baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _configure_run_id_from_cli() -> None:
    """Set the run ID before importing config, whose paths are import-time values."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id", default=None)
    args, _ = parser.parse_known_args()
    if args.run_id:
        os.environ["KG_TABI_RUN_ID"] = args.run_id


_configure_run_id_from_cli()

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src import controlled_benchmark as benchmark
from src.graph_analysis import build_graph, louvain_partition


REPORT_VERSION = "1.0"
TARGETED_CONDITION = "targeted_boundary"
CONTROL_CONDITIONS = ("internal_edge_control", "global_random_edge_control")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report_path(path: Path) -> str:
    """Prefer a portable repository-relative path in the written artifact."""
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}.")
            records.append(value)
    return records


def _deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one record per target-condition-level-selected-mask instance."""
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record.get("perturbation_key")
        if not isinstance(key, str):
            raise ValueError("Trial record is missing its perturbation_key.")
        unique.setdefault(key, record)
    return list(unique.values())


def _summary_rows(result: dict[str, Any], mode: str, condition: str) -> list[dict[str, Any]]:
    try:
        rows = result["summary"][mode]
    except KeyError as exc:
        raise ValueError(f"Benchmark summary is missing summary.{mode}.") from exc
    if not isinstance(rows, list):
        raise ValueError(f"Benchmark summary field summary.{mode} must be a list.")
    return sorted(
        (row for row in rows if row.get("condition") == condition),
        key=lambda row: float(row["mask_level"]),
    )


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    return float(value)


def _interval(metrics: dict[str, Any], name: str) -> list[float] | None:
    interval = metrics.get("metric_bootstrap_95_ci", {}).get(name)
    if interval is None:
        return None
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"Invalid bootstrap interval for {name}.")
    return [float(interval[0]), float(interval[1])]


def _metric_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["primary_unique_perturbation_metrics"]
    return {
        "mask_level": float(row["mask_level"]),
        "unique_target_mask_instances": int(row["unique_perturbations"]),
        "nominal_seeded_trials": int(row["nominal_seeded_trials"]),
        "true_positives": int(metrics["true_positives"]),
        "false_positives": int(metrics["false_positives"]),
        "false_negatives": int(metrics["false_negatives"]),
        "detected_orphan_outputs": int(metrics["predicted_clusters"]),
        "precision": _metric_value(metrics, "precision"),
        "precision_bootstrap_95_ci": _interval(metrics, "precision"),
        "recall": _metric_value(metrics, "recall"),
        "recall_bootstrap_95_ci": _interval(metrics, "recall"),
        "f1": _metric_value(metrics, "f1"),
        "f1_bootstrap_95_ci": _interval(metrics, "f1"),
    }


def _control_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["primary_unique_perturbation_metrics"]
    output_interval = metrics["any_output_wilson_95_ci"]
    return {
        "mask_level": float(row["mask_level"]),
        "unique_perturbations": int(row["unique_perturbations"]),
        "nominal_seeded_trials": int(row["nominal_seeded_trials"]),
        "detected_orphan_outputs": int(metrics["predicted_clusters"]),
        "false_positive_outputs": int(metrics["false_positives"]),
        "any_output_rate": float(metrics["any_output_rate"]),
        "any_output_wilson_95_ci": [float(output_interval[0]), float(output_interval[1])],
    }


def _validate_primary_metrics(result: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate accounting identities for the report's analysis unit."""
    trial_counts: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in trials:
        trial_counts[(str(record["condition"]), float(record["mask_level"]))].append(record)

    checked_groups = 0
    for mode in ("metric_only", "end_to_end"):
        for condition in (TARGETED_CONDITION, *CONTROL_CONDITIONS):
            for row in _summary_rows(result, mode, condition):
                key = (condition, float(row["mask_level"]))
                records = trial_counts[key]
                if len(records) != int(row["nominal_seeded_trials"]):
                    raise ValueError(f"Seeded-trial count mismatch for {mode}, {key}.")
                if len(_deduplicate(records)) != int(row["unique_perturbations"]):
                    raise ValueError(f"Unique-perturbation count mismatch for {mode}, {key}.")

                metrics = row["primary_unique_perturbation_metrics"]
                tp = int(metrics["true_positives"])
                fp = int(metrics["false_positives"])
                fn = int(metrics["false_negatives"])
                outputs = int(metrics["predicted_clusters"])
                if tp + fp != outputs:
                    raise ValueError(f"TP + FP does not equal output count for {mode}, {key}.")
                if condition == TARGETED_CONDITION:
                    if tp + fn != int(row["unique_perturbations"]):
                        raise ValueError(f"TP + FN does not equal target-mask units for {mode}, {key}.")
                elif tp != 0 or fn != 0 or fp != outputs:
                    raise ValueError(f"Control accounting is invalid for {mode}, {key}.")
                checked_groups += 1

    return {
        "summary_groups_checked": checked_groups,
        "nominal_seeded_trials": len(trials),
        "unique_perturbations": len(_deduplicate(trials)),
        "status": "passed",
    }


def _conductance(undirected_graph: Any, nodes: frozenset[Any]) -> float:
    """Return standard unweighted conductance for a nontrivial node set."""
    _, cut_edges, _ = benchmark._bridge_statistics(undirected_graph, nodes)
    volume_inside = sum(undirected_graph.degree(node) for node in nodes)
    outside = set(undirected_graph.nodes()) - set(nodes)
    volume_outside = sum(undirected_graph.degree(node) for node in outside)
    denominator = min(volume_inside, volume_outside)
    return cut_edges / denominator if denominator else 0.0


def _normalized_cut(undirected_graph: Any, nodes: frozenset[Any]) -> float:
    """Return the standard unweighted two-way normalized-cut value.

    ``Ncut(S, not-S) = cut(S, not-S)/vol(S) + cut(S, not-S)/vol(not-S)``.
    The benchmark candidate communities are nontrivial, but an infinite value
    is safer than silently treating a zero-volume degenerate set as isolated.
    """
    _, cut_edges, _ = benchmark._bridge_statistics(undirected_graph, nodes)
    volume_inside = sum(undirected_graph.degree(node) for node in nodes)
    outside = set(undirected_graph.nodes()) - set(nodes)
    volume_outside = sum(undirected_graph.degree(node) for node in outside)
    if volume_inside <= 0 or volume_outside <= 0:
        return float("inf")
    return cut_edges / volume_inside + cut_edges / volume_outside


def _random_score_expectation(candidate_count: int) -> dict[str, float]:
    """Exact chance expectation for iid continuous scores over a fixed candidate set."""
    if candidate_count < 2:
        raise ValueError("A random-score reference requires at least two candidate communities.")
    return {
        "conditional_pairwise_auc": 0.5,
        "tie_aware_top1_rate": 1.0 / candidate_count,
    }


def _tie_aware_pairwise_auc(scores: list[tuple[float, bool]]) -> float:
    """Probability that a positive score exceeds a negative score, ties worth 1/2."""
    positives = [score for score, label in scores if label]
    negatives = [score for score, label in scores if not label]
    if not positives or not negatives:
        raise ValueError("AUC requires both positive and negative candidate instances.")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _tie_aware_top1(scores: list[tuple[float, bool]]) -> float:
    """Give a tied top rank equal fractional credit to each tied candidate."""
    maximum = max(score for score, _ in scores)
    top = [(score, label) for score, label in scores if score == maximum]
    return sum(1.0 for _, label in top if label) / len(top)


def _comparison_for_level(
    records: list[dict[str, Any]],
    graph: Any,
    baseline_by_id: dict[int, dict[str, Any]],
    edge_groups: dict[tuple[Any, Any], tuple[tuple[Any, Any], ...]],
) -> dict[str, Any]:
    """Compare threshold-free scores on the fixed-partition candidate universe."""
    all_edges = sorted(
        edge_groups,
        key=lambda edge: (benchmark._node_key(edge[0]), benchmark._node_key(edge[1])),
    )
    score_pairs: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    top1_total: dict[str, float] = defaultdict(float)
    target_count = 0
    candidate_score_instances = 0
    conductance_monotonic_transform_instances = 0

    for record in records:
        target_id = int(record["target"]["baseline_community_id"])
        target = baseline_by_id.get(target_id)
        if target is None:
            raise ValueError(f"Target community {target_id} is not in the eligible candidate universe.")
        boundary_pool = [
            edge
            for edge in all_edges
            if (edge[0] in target["nodes"]) != (edge[1] in target["nodes"])
        ]
        selected = benchmark._sample_edges(
            boundary_pool,
            int(record["removed_undirected_edges"]),
            "boundary",
            target_id,
            float(record["mask_level"]),
            int(record["masking_seed"]),
        )
        signature = benchmark._edge_signature(selected)
        if signature != record["mask_signature_sha256"]:
            raise ValueError("Cannot reproduce a targeted boundary mask from its trial metadata.")
        masked_graph, _ = benchmark._mask_graph(graph, edge_groups, selected)
        undirected = masked_graph.to_undirected()
        by_score: dict[str, list[tuple[float, bool]]] = {
            "bridge_ratio": [],
            "conductance": [],
            "normalized_cut": [],
            "community_size": [],
        }
        for community_id, candidate in baseline_by_id.items():
            internal_edges, cut_edges, bridge_ratio = benchmark._bridge_statistics(
                undirected, candidate["nodes"]
            )
            conductance = _conductance(undirected, candidate["nodes"])
            normalized_cut = _normalized_cut(undirected, candidate["nodes"])
            volume_inside = 2 * internal_edges + cut_edges
            volume_outside = sum(
                undirected.degree(node) for node in set(undirected.nodes()) - set(candidate["nodes"])
            )
            candidate_score_instances += 1
            if volume_inside <= volume_outside:
                conductance_monotonic_transform_instances += 1
            # Higher values mean "more likely to be the induced isolated target".
            by_score["bridge_ratio"].append((-bridge_ratio, community_id == target_id))
            by_score["conductance"].append((-conductance, community_id == target_id))
            by_score["normalized_cut"].append((-normalized_cut, community_id == target_id))
            by_score["community_size"].append((float(candidate["size"]), community_id == target_id))
        for name, scores in by_score.items():
            score_pairs[name].extend(scores)
            top1_total[name] += _tie_aware_top1(scores)
        target_count += 1

    score_results: dict[str, dict[str, float]] = {}
    for name, scores in score_pairs.items():
        score_results[name] = {
            "conditional_pairwise_auc": round(_tie_aware_pairwise_auc(scores), 8),
            "tie_aware_top1_rate": round(top1_total[name] / target_count, 8),
        }

    size_only_tp = target_count
    size_only_fp = target_count * (len(baseline_by_id) - 1)
    size_only_precision = size_only_tp / (size_only_tp + size_only_fp)
    return {
        "unique_target_mask_instances": target_count,
        "candidate_communities_after_fixed_size_gate": len(baseline_by_id),
        "candidate_score_instances": candidate_score_instances,
        "conductance_monotonic_transform_instances": conductance_monotonic_transform_instances,
        "threshold_free_rank_scores": score_results,
        "random_score_expectation": _random_score_expectation(len(baseline_by_id)),
        "size_only_reference": {
            "rule": "Return every community in the fixed 5%-size-gated candidate universe.",
            "true_positives": size_only_tp,
            "false_positives": size_only_fp,
            "false_negatives": 0,
            "precision": round(size_only_precision, 8),
            "recall": 1.0,
            "f1": round(2 * size_only_precision / (1 + size_only_precision), 8),
        },
    }


def _structural_comparison(
    result: dict[str, Any], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run only a threshold-free, fixed-membership supplemental comparison."""
    resolved_path = Path(str(result["run"]["resolved_triples_path"]))
    if not resolved_path.exists():
        raise FileNotFoundError(f"Resolved triples artifact is unavailable: {resolved_path}")
    if _sha256_file(resolved_path) != result["run"]["resolved_triples_sha256"]:
        raise ValueError("Resolved triples hash differs from the benchmark source artifact.")
    with resolved_path.open(encoding="utf-8") as handle:
        graph = build_graph(json.load(handle))
    baseline_partition = louvain_partition(graph)
    eligible = benchmark._eligible_baseline_communities(graph, baseline_partition)
    expected_ids = {
        int(row["community_id"]) for row in result["baseline"]["eligible_communities"]
    }
    baseline_by_id = {int(row["id"]): row for row in eligible if int(row["id"]) in expected_ids}
    if set(baseline_by_id) != expected_ids:
        raise ValueError("The fixed-partition eligible community set does not match the benchmark summary.")
    edge_groups = benchmark._undirected_edge_groups(graph)
    targeted = [record for record in trials if record["condition"] == TARGETED_CONDITION]
    by_level: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in _deduplicate(targeted):
        by_level[float(record["mask_level"])].append(record)

    rows = [
        {
            "mask_level": level,
            **_comparison_for_level(records, graph, baseline_by_id, edge_groups),
        }
        for level, records in sorted(by_level.items())
    ]
    comparison_instances = sum(row["candidate_score_instances"] for row in rows)
    monotonic_instances = sum(row["conductance_monotonic_transform_instances"] for row in rows)
    return {
        "status": "implemented_as_metric_only_threshold_free_comparison",
        "measurement_scope": (
            "The baseline Louvain membership is fixed; each perturbation has one induced "
            "positive among the three prequalified communities. These are descriptive "
            "rank comparisons, not calibrated end-to-end detector tests."
        ),
        "post_hoc_status": (
            "These supplemental reference rankings were added after reviewer feedback. "
            "They use no fitted cutoff or tuned parameter and do not alter the benchmark's "
            "primary end-to-end results."
        ),
        "candidate_universe": {
            "rule": "Communities satisfying the pre-specified 5% size gate in the baseline partition.",
            "count": len(baseline_by_id),
            "community_ids": sorted(baseline_by_id),
        },
        "score_definitions": {
            "bridge_ratio": (
                "Rank by negative R_bridge, where R_bridge = cut(S, not-S) / "
                "(internal(S) + cut(S, not-S)); lower is more isolated."
            ),
            "conductance": (
                "Rank by negative phi(S), where phi(S) = cut(S, not-S) / "
                "min(vol(S), vol(not-S)); lower is more isolated."
            ),
            "normalized_cut": (
                "Rank by negative Ncut(S, not-S), where Ncut = cut(S, not-S)/vol(S) + "
                "cut(S, not-S)/vol(not-S); lower is more isolated."
            ),
            "community_size": "Rank by fixed community size only; higher is treated as more likely.",
            "external_edge_fraction": (
                "Not an independent comparator here: the unweighted external-adjacency fraction "
                "with denominator internal + external is exactly R_bridge."
            ),
        },
        "observed_bridge_conductance_relation": (
            "For every evaluated candidate score instance, vol(S) <= vol(not-S). Therefore "
            "conductance equals R_bridge / (2 - R_bridge), a monotonic transform of the "
            "bridge ratio; identical rank results are expected in this fixed candidate universe."
            if comparison_instances == monotonic_instances
            else (
                "Conductance and bridge ratio are distinct scores in this candidate universe; "
                "their rank results should not be interpreted as algebraically identical."
            )
        ),
        "bridge_conductance_monotonic_transform_coverage": {
            "candidate_score_instances": comparison_instances,
            "instances_with_vol_s_not_exceeding_vol_complement": monotonic_instances,
        },
        "why_no_conductance_f1": (
            "Neither a conductance nor normalized-cut cutoff was pre-specified or calibrated "
            "on an independent development set. Selecting either after inspecting these masks "
            "would create an optimistic comparison, so both are reported only as threshold-free "
            "rank scores."
        ),
        "random_score_reference": (
            "The random-score reference is the exact expectation for iid continuous scores "
            "independent of the candidate and perturbation: pairwise AUC = 0.5 and top-1 = "
            "1/K for K candidates. An arbitrary seeded random draw is intentionally omitted "
            "because it would add Monte-Carlo noise without testing a different detector."
        ),
        "why_no_end_to_end_baseline_claim": (
            "After Louvain is rerun, a baseline community can split, merge, or disappear; a "
            "fair conductance detector would require a separately specified community proposal "
            "stage and a held-out calibration protocol."
        ),
        "rows": rows,
    }


def _format_number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _format_interval(interval: list[float] | None, digits: int = 3) -> str:
    if interval is None:
        return "n/a"
    return f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def _metric_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Masked boundary adjacencies | n | TP | FP | FN | Precision [95% bootstrap CI] | Recall [95% bootstrap CI] | F1 [95% bootstrap CI] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def with_interval(value: float | None, interval: list[float] | None) -> str:
            formatted = _format_number(value)
            return formatted if interval is None else f"{formatted} {_format_interval(interval)}"

        lines.append(
            "| {level:.0%} | {n} | {tp} | {fp} | {fn} | {p} | {r} | {f1} |".format(
                level=row["mask_level"],
                n=row["unique_target_mask_instances"],
                tp=row["true_positives"],
                fp=row["false_positives"],
                fn=row["false_negatives"],
                p=with_interval(row["precision"], row["precision_bootstrap_95_ci"]),
                r=with_interval(row["recall"], row["recall_bootstrap_95_ci"]),
                f1=with_interval(row["f1"], row["f1_bootstrap_95_ci"]),
            )
        )
    return "\n".join(lines)


def _control_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Control | Mask level | n | FP outputs | Any-output rate [95% Wilson CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {level:.0%} | {n} | {fp} | {rate} {interval} |".format(
                condition=row["condition"],
                level=row["mask_level"],
                n=row["unique_perturbations"],
                fp=row["false_positive_outputs"],
                rate=_format_number(row["any_output_rate"]),
                interval=_format_interval(row["any_output_wilson_95_ci"]),
            )
        )
    return "\n".join(lines)


def _comparison_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Mask level | n | Bridge-ratio AUC / top-1 | Conductance AUC / top-1 | Normalized-cut AUC / top-1 | Size-only AUC / top-1 | Random-score expected AUC / top-1 | Size-only P/R/F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        scores = row["threshold_free_rank_scores"]
        size = row["size_only_reference"]
        lines.append(
            "| {level:.0%} | {n} | {bridge_auc:.3f} / {bridge_top:.3f} | {cond_auc:.3f} / {cond_top:.3f} | {ncut_auc:.3f} / {ncut_top:.3f} | {size_auc:.3f} / {size_top:.3f} | {random_auc:.3f} / {random_top:.3f} | {p:.3f} / {r:.3f} / {f1:.3f} |".format(
                level=row["mask_level"],
                n=row["unique_target_mask_instances"],
                bridge_auc=scores["bridge_ratio"]["conditional_pairwise_auc"],
                bridge_top=scores["bridge_ratio"]["tie_aware_top1_rate"],
                cond_auc=scores["conductance"]["conditional_pairwise_auc"],
                cond_top=scores["conductance"]["tie_aware_top1_rate"],
                ncut_auc=scores["normalized_cut"]["conditional_pairwise_auc"],
                ncut_top=scores["normalized_cut"]["tie_aware_top1_rate"],
                size_auc=scores["community_size"]["conditional_pairwise_auc"],
                size_top=scores["community_size"]["tie_aware_top1_rate"],
                random_auc=row["random_score_expectation"]["conditional_pairwise_auc"],
                random_top=row["random_score_expectation"]["tie_aware_top1_rate"],
                p=size["precision"],
                r=size["recall"],
                f1=size["f1"],
            )
        )
    return "\n".join(lines)


def _render_markdown(report: dict[str, Any]) -> str:
    end_to_end = report["primary_end_to_end"]
    metric_only = report["metric_only"]
    controls = report["negative_controls"]
    comparison = report["structural_reference_comparison"]
    return "\n".join(
        [
            "# Controlled Perturbation Benchmark: Appendix Report",
            "",
            f"Artifact version: `{report['report_version']}`  ",
            f"Run: `{report['source']['run_id']}`  ",
            f"Input summary SHA-256: `{report['source']['summary_sha256']}`  ",
            f"Input trials SHA-256: `{report['source']['trials_sha256']}`",
            "",
            "## Scope",
            "",
            "This is a conditional, offline structural-isolation perturbation check of the topology-stage orphan detector. It does not evaluate TABI generation, scientific novelty, evidence support, or research-gap usefulness.",
            "",
            "## Analysis unit and labeling",
            "",
            "The primary unit is one **target-community by unique selected-edge-mask instance**. Repeated seeds that produce the same selected edge set for the same target, condition, and mask level are deduplicated before the primary summaries and bootstrap intervals. Each targeted-boundary instance induces exactly one positive: its deliberately boundary-masked baseline community. After one-to-one Jaccard matching at the pre-specified threshold, a TP is one output matched to that target; an FN is a target with no such match; and every remaining orphan output is an FP, including an output matched to a non-target baseline community. Metrics pool these event units across unique instances at a mask level. A precision or F1 value is `n/a` when there are no outputs; in negative controls, P/R/F1 are not defined because there are no positive instances.",
            "",
            "The intervals below are 2,000-resample percentile bootstraps over unique perturbation instances. They describe this fixed graph and are not population-generalization intervals.",
            "",
            "## Primary end-to-end topology-stage measurement",
            "",
            _metric_markdown_table(end_to_end),
            "",
            "End-to-end reruns Louvain after masking and then applies the orphan detector and matching procedure.",
            "",
            "## Fixed-membership metric-only measurement",
            "",
            _metric_markdown_table(metric_only),
            "",
            "Metric-only retains the baseline Louvain membership and reapplies the size/bridge-ratio rule after masking; it isolates the detector metric from partition changes.",
            "",
            "## Negative controls",
            "",
            "Both controls contain no induced positive. `internal_edge_control` removes the same number of within-target adjacencies while preserving the target boundary; `global_random_edge_control` removes the same number of graph adjacencies without targeting a community boundary. Reported FP outputs are all outputs in these no-positive trials.",
            "",
            "### End-to-end controls",
            "",
            _control_markdown_table(controls["end_to_end"]),
            "",
            "### Metric-only controls",
            "",
            _control_markdown_table(controls["metric_only"]),
            "",
            "## Threshold-free structural reference comparisons (metric-only only)",
            "",
            comparison["measurement_scope"],
            "",
            comparison["post_hoc_status"],
            "",
            _comparison_markdown_table(comparison["rows"]),
            "",
            "AUC is the within-perturbation, tie-aware probability that the induced target receives a higher score than an eligible negative; top-1 gives fractional credit for ties. The random-score column is the analytic iid-continuous-score expectation, not a noisy seed-dependent draw. `external-edge fraction` is not shown as a separate method because, with the unweighted internal-plus-external denominator used here, it is exactly the production bridge ratio. The size-only reference outputs all three size-gated communities, so it has fixed P=1/3, R=1, F1=1/2.",
            "",
            comparison["observed_bridge_conductance_relation"],
            "",
            comparison["random_score_reference"],
            "",
            comparison["why_no_conductance_f1"],
            "",
            comparison["why_no_end_to_end_baseline_claim"],
            "",
            "## Validation",
            "",
            "The generator verifies source hashes, seeded and deduplicated trial counts, TP+FP=output count, TP+FN=number of induced-positive units, control accounting, and reconstruction of every unique targeted mask from its recorded metadata.",
        ]
    ) + "\n"


def build_report(
    summary_path: Path | None = None,
    trials_path: Path | None = None,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Build and write JSON and Markdown appendix artifacts."""
    summary_path = summary_path or Path(config.DATA_DIR) / "controlled_benchmark_multicondition_results.json"
    trials_path = trials_path or Path(config.DATA_DIR) / "controlled_benchmark_multicondition_trials.jsonl"
    output_directory = output_directory or Path(config.DATA_DIR)
    result = _read_json(summary_path)
    trials = _read_jsonl(trials_path)
    validation = _validate_primary_metrics(result, trials)
    comparison = _structural_comparison(result, trials)
    validation["targeted_unique_masks_reconstructed"] = sum(
        row["unique_target_mask_instances"] for row in comparison["rows"]
    )

    controls: dict[str, list[dict[str, Any]]] = {}
    for mode in ("end_to_end", "metric_only"):
        controls[mode] = [
            {"condition": condition, **_control_row(row)}
            for condition in CONTROL_CONDITIONS
            for row in _summary_rows(result, mode, condition)
        ]

    report = {
        "report_version": REPORT_VERSION,
        "scope": result["scope"],
        "source": {
            "run_id": result["run"].get("run_id"),
            "summary_path": _report_path(summary_path),
            "summary_sha256": _sha256_file(summary_path),
            "trials_path": _report_path(trials_path),
            "trials_sha256": _sha256_file(trials_path),
            "resolved_triples_sha256": result["run"]["resolved_triples_sha256"],
        },
        "primary_analysis_unit": {
            "unit": "one target-community by unique selected-edge-mask instance",
            "deduplication": (
                "Repeated masking seeds that select the same edge set for the same target, "
                "condition, and mask level count once in primary summaries."
            ),
            "targeted_positive_definition": "The deliberately boundary-masked target community.",
            "true_positive": "One output one-to-one Jaccard-matched to the induced target at J >= 0.80.",
            "false_negative": "An induced target without such a matched output.",
            "false_positive": (
                "Any other orphan output, including a match to a non-target baseline community "
                "or an output unmatched to every baseline community."
            ),
            "undefined_metric_convention": (
                "Precision and F1 are undefined when there are no outputs. All P/R/F1 metrics "
                "are undefined for no-positive controls."
            ),
        },
        "interval_method": result["design"]["intervals"],
        "primary_end_to_end": [
            _metric_row(row) for row in _summary_rows(result, "end_to_end", TARGETED_CONDITION)
        ],
        "metric_only": [
            _metric_row(row) for row in _summary_rows(result, "metric_only", TARGETED_CONDITION)
        ],
        "negative_controls": controls,
        "structural_reference_comparison": comparison,
        "validation": validation,
        "limitations": [
            "The perturbations, targets, and candidate communities all come from one fixed legacy graph.",
            "The comparison is a conditional structural test and cannot establish real-world gap-detection accuracy.",
            "A calibrated conductance detector, an end-to-end community proposal baseline, and an external held-out corpus remain future work.",
        ],
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "controlled_benchmark_appendix_report.json"
    markdown_path = output_directory / "controlled_benchmark_appendix_report.md"
    report["artifacts"] = {
        "json": _report_path(json_path),
        "markdown": _report_path(markdown_path),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write(_render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=config.RUN_ID or None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--trials-path", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    args = parser.parse_args()
    report = build_report(args.summary_path, args.trials_path, args.output_directory)
    print(f"[+] Wrote {report['artifacts']['json']}")
    print(f"[+] Wrote {report['artifacts']['markdown']}")
    print(
        "[+] Validated "
        f"{report['validation']['summary_groups_checked']} summary groups and "
        f"{report['validation']['targeted_unique_masks_reconstructed']} unique targeted masks."
    )


if __name__ == "__main__":
    main()
