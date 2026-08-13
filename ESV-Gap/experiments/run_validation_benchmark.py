"""Controlled evaluation for the revised ESV-Gap evidence contract.

Development and test seeds are disjoint.  The test generator covers all three
signal families and includes negatives that are not simple copies of one hard
rule.  Results remain an internal validity study, not expert evidence of
scientific novelty or usefulness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validate_gaps import validate_candidate


BINARY_MAPPING = {
    "ground_truth_positive": "label == 1: planted evidence-sufficient candidate",
    "ground_truth_negative": "label == 0: planted artefact or covered/qualified candidate",
    "prediction_positive": "validation status == automatically_eligible",
    "prediction_negative": "validation status in {review_required, rejected}",
}


def binary_prediction(decision: dict) -> int:
    """Map the three-way gate to the benchmark's explicit binary outcome."""
    return int(decision["status"] == "automatically_eligible")


def config(
    seed: int,
    edge_keep: float = 0.80,
    repeats: int = 100,
    stability_threshold: float = 0.70,
) -> dict:
    return {
        "gap_validation": {
            "min_supporting_papers": 2,
            "min_independent_paths": 2,
            "min_surviving_paths": 1,
            "max_path_length": 4,
            "min_specificity": 0.55,
            "bootstrap_repeats": repeats,
            "edge_keep_probability": edge_keep,
            "paper_keep_probability": edge_keep,
            "plausible_edge_add_probability": 0.50,
            "min_stability": stability_threshold,
            "orphan_isolation_threshold": 0.90,
            "temporal_decay_threshold": 0.30,
            "temporal_lookback_years": 2,
            "closure_token_coverage": 0.60,
            "random_seed": seed,
            "max_closure_hits_to_record": 10,
            "weights": {
                "provenance": 0.25,
                "specificity": 0.15,
                "path_diversity": 0.20,
                "stability": 0.30,
                "closure_clearance": 0.10,
            },
        }
    }


def add_path(graph, head, middle, tail, paper_a, paper_b):
    graph.add_node(head, papers=[paper_a])
    graph.add_node(middle, papers=[paper_a, paper_b])
    graph.add_node(tail, papers=[paper_b])
    graph.add_edge(head, middle, source_paper=paper_a, year=2022)
    graph.add_edge(middle, tail, source_paper=paper_b, year=2024)


def missing_link_cases(seed: int):
    suffix = str(seed)
    head = f"runtime anomaly detection {suffix}"
    tail = f"service mesh telemetry {suffix}"
    base = nx.MultiDiGraph()
    add_path(base, head, f"distributed tracing {suffix}", tail, f"p{suffix}a", f"p{suffix}b")
    add_path(base, head, f"policy enforcement {suffix}", tail, f"p{suffix}c", f"p{suffix}d")
    candidate = {"type": "missing_link", "head": head, "tail": tail, "detector_score": 0.74}
    yield base, candidate, [], 1, "supported_two_path", "missing_link"

    short = nx.MultiDiGraph()
    add_path(short, "mTLS", f"zero trust gateway {suffix}", "JWT", f"s{suffix}a", f"s{suffix}b")
    add_path(short, "mTLS", f"identity broker {suffix}", "JWT", f"s{suffix}c", f"s{suffix}d")
    yield short, {"type": "missing_link", "head": "mTLS", "tail": "JWT", "detector_score": 0.70}, [], 1, "short_valid_entities", "missing_link"

    existing = base.copy()
    existing.add_edge(head, tail, source_paper=f"p{suffix}x", year=2025, relation="EVALUATES")
    yield existing, {**candidate, "relation": "IMPROVES", "detector_score": 0.91}, [], 0, "qualified_existing_relation", "missing_link"

    generic = nx.MultiDiGraph()
    add_path(generic, "proposed framework", f"component {suffix}", "algorithm", f"g{suffix}a", f"g{suffix}b")
    add_path(generic, "proposed framework", f"module {suffix}", "algorithm", f"g{suffix}c", f"g{suffix}d")
    yield generic, {"type": "missing_link", "head": "proposed framework", "tail": "algorithm", "detector_score": 0.88}, [], 0, "generic_entity", "missing_link"

    fragile = nx.MultiDiGraph()
    add_path(fragile, head, f"fragile bridge {suffix}", tail, f"f{suffix}a", f"f{suffix}b")
    yield fragile, candidate, [], 0, "single_path", "missing_link"

    shared = nx.MultiDiGraph()
    add_path(shared, head, f"shared bridge a {suffix}", tail, f"shared{suffix}", f"a{suffix}")
    add_path(shared, head, f"shared bridge b {suffix}", tail, f"shared{suffix}", f"b{suffix}")
    yield shared, candidate, [], 0, "source_dependent_paths", "missing_link"

    addition = base.copy()
    plausible = {**candidate, "plausible_edges": [{"head": head, "tail": tail, "paper_id": f"low{suffix}"}]}
    yield addition, plausible, [], 0, "plausible_missing_edge", "missing_link"

    synonym_docs = [{
        "paper_id": f"closure{suffix}",
        "title": "Runtime anomaly monitoring with service-mesh observability",
        "abstract": "The study evaluates their relationship in production.",
        "year": 2025,
    }]
    synonym = {
        **candidate,
        "entity_aliases": {
            head: ["runtime anomaly monitoring"],
            tail: ["service mesh observability"],
        },
    }
    yield base, synonym, synonym_docs, 0, "synonym_coverage", "missing_link"

    typed = nx.MultiDiGraph()
    add_path(typed, "JWT [METHOD]", f"gateway audit {suffix}", "OAuth", f"v{suffix}a", f"v{suffix}b")
    add_path(typed, "JWT [METHOD]", f"token policy {suffix}", "OAuth", f"v{suffix}c", f"v{suffix}d")
    typed_docs = [{
        "paper_id": f"typed-closure{suffix}",
        "title": "Applying OAuth2 and JWT protocols",
        "abstract": "OAuth2 and JWT secure distributed API gateways.",
        "year": 2025,
    }]
    yield typed, {
        "type": "missing_link",
        "head": "JWT [METHOD]",
        "tail": "OAuth",
        "detector_score": 0.89,
    }, typed_docs, 0, "typed_suffix_coverage", "missing_link"

    self_link = nx.MultiDiGraph()
    add_path(self_link, "Scalability [CONCEPT]", f"capacity bridge {suffix}", "scalability [METRIC]", f"c{suffix}a", f"c{suffix}b")
    add_path(self_link, "Scalability [CONCEPT]", f"performance bridge {suffix}", "scalability [METRIC]", f"c{suffix}c", f"c{suffix}d")
    yield self_link, {
        "type": "missing_link",
        "head": "Scalability [CONCEPT]",
        "tail": "scalability [METRIC]",
        "detector_score": 0.93,
    }, [], 0, "canonical_self_link", "missing_link"


def orphan_graph(seed: int, shared_paper: bool = False):
    graph = nx.MultiDiGraph()
    members = [f"federated policy concept {seed}-{i}" for i in range(5)]
    index = 0
    for i, head in enumerate(members):
        for tail in members[i + 1:]:
            paper = f"community-{seed}" if shared_paper else f"community-{seed}-{index}"
            graph.add_edge(head, tail, source_paper=paper, year=2023 + index % 2)
            index += 1
    graph.add_edge(members[0], f"external security concept {seed}", source_paper=f"bridge-{seed}", year=2024)
    candidate = {
        "type": "orphan_cluster",
        "community_id": seed,
        "members": members,
        "key_concepts": members,
        "detector_score": 0.68,
    }
    return graph, candidate


def orphan_cases(seed: int):
    graph, candidate = orphan_graph(seed)
    yield graph, candidate, [], 1, "stable_specific_community", "orphan_cluster"

    dependent, dependent_candidate = orphan_graph(seed + 10_000, shared_paper=True)
    yield dependent, dependent_candidate, [], 0, "single_study_community", "orphan_cluster"

    bridged, bridged_candidate = orphan_graph(seed + 20_000)
    outsiders = [f"outside topic {seed}-{i}" for i in range(5)]
    bridged_candidate = {
        **bridged_candidate,
        "plausible_edges": [
            {"head": bridged_candidate["members"][i], "tail": outsiders[i], "paper_id": f"low-bridge-{seed}-{i}"}
            for i in range(5)
        ],
        "detector_score": 0.84,
    }
    yield bridged, bridged_candidate, [], 0, "plausible_bridge_additions", "orphan_cluster"


def temporal_graph(seed: int, counts: dict[int, int], concept: str):
    graph = nx.MultiDiGraph()
    for year, count in counts.items():
        for index in range(count):
            paper = f"t-{seed}-{year}-{index}"
            graph.add_edge(concept, f"temporal evidence {seed}-{year}-{index}", source_paper=paper, year=year)
    return graph


def temporal_cases(seed: int):
    totals = {2022: 20, 2023: 20, 2024: 100, 2025: 100, 2026: 40}
    concept = f"post-quantum service authentication {seed}"
    positive = temporal_graph(seed, {2022: 4, 2023: 4, 2024: 3, 2025: 3}, concept)
    candidate = {
        "type": "temporal_decay",
        "concept": concept,
        "peak_year": 2022,
        "publication_counts": totals,
        "analysis_end_year": 2025,
        "detector_score": 0.72,
    }
    yield positive, candidate, [], 1, "normalised_decline", "temporal_decay"

    stable = temporal_graph(seed + 30_000, {2022: 4, 2023: 4, 2024: 20, 2025: 20}, concept)
    yield stable, candidate, [], 0, "raw_decline_but_rate_stable", "temporal_decay"

    censored = temporal_graph(seed + 40_000, {2022: 4, 2023: 4, 2024: 20, 2025: 20, 2026: 1}, concept)
    censored_candidate = {**candidate, "peak_year": 2025}
    yield censored, censored_candidate, [], 0, "partial_final_year", "temporal_decay"


def benchmark_cases(seed: int):
    # The controlled fixture models a completed candidate-level plausible-edge
    # search.  Empty pools therefore mean "searched, no relevant edge found";
    # production candidates without this marker fail closed to manual review.
    for case in (*missing_link_cases(seed), *orphan_cases(seed), *temporal_cases(seed)):
        graph, candidate, documents, label, case_type, signal_type = case
        yield (
            graph,
            {**candidate, "plausible_edge_search_performed": True},
            documents,
            label,
            case_type,
            signal_type,
        )


def metrics(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def ranking_metrics(labels, scores, k):
    order = sorted(range(len(labels)), key=lambda i: (-scores[i], i))
    ranked = [labels[i] for i in order]
    k = min(k, len(ranked))
    precision_at_k = sum(ranked[:k]) / max(k, 1)
    recall_at_k = sum(ranked[:k]) / max(sum(labels), 1)
    dcg = sum(label / math.log2(index + 2) for index, label in enumerate(ranked[:k]))
    ideal = sorted(labels, reverse=True)
    idcg = sum(label / math.log2(index + 2) for index, label in enumerate(ideal[:k]))
    hits = 0
    precisions = []
    for rank, label in enumerate(ranked, 1):
        if label:
            hits += 1
            precisions.append(hits / rank)
    return {
        f"precision@{k}": round(precision_at_k, 4),
        f"recall@{k}": round(recall_at_k, 4),
        f"ndcg@{k}": round(dcg / idcg if idcg else 0.0, 4),
        "average_precision": round(sum(precisions) / max(sum(labels), 1), 4),
    }


def _prediction_for_ablation(name, decision):
    metrics_ = decision["metrics"]
    reasons = set(decision["reasons"])
    checks = {
        "provenance_only": metrics_["provenance"] >= 1.0,
        "specificity_only": metrics_["specificity"] >= 0.55,
        "path_diversity_only": metrics_["path_diversity"] >= 1.0,
        "perturbation_only": metrics_["stability"] >= 0.70,
        "hard_contract_without_perturbation": not reasons.intersection({
            "insufficient_independent_paper_support",
            "generic_or_underspecified_entities",
            "insufficient_source_disjoint_evidence_paths",
            "observed_relation_requires_qualified_review",
            "canonical_self_link",
        }),
        "full_without_coverage_screen": not reasons.intersection({
            "insufficient_independent_paper_support",
            "generic_or_underspecified_entities",
            "insufficient_source_disjoint_evidence_paths",
            "observed_relation_requires_qualified_review",
            "unstable_under_multi_mode_perturbation",
            "canonical_self_link",
        }),
        "full_evidence_contract": bool(binary_prediction(decision)),
    }
    return int(checks[name])


def evaluate(seed_start: int, repetitions: int, cfg_factory=config, case_filter=None):
    rows = []
    for seed in range(seed_start, seed_start + repetitions):
        for graph, candidate, documents, label, case_type, signal_type in benchmark_cases(seed):
            if case_filter is not None and case_type not in case_filter:
                continue
            decision = validate_candidate(graph, candidate, cfg_factory(seed), documents)
            rows.append({
                "seed": seed,
                "case_type": case_type,
                "signal_type": signal_type,
                "label": label,
                "detector_score": candidate.get("detector_score", 0.0),
                "prediction": binary_prediction(decision),
                "status": decision["status"],
                "ranking_score": decision["ranking_score"],
                "metrics": decision["metrics"],
                "mode_survival": decision["bootstrap"]["mode_survival"],
                "reasons": decision["reasons"],
            })
    return rows


def grouped_results(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {
        name: metrics(
            [row["label"] for row in items],
            [row["prediction"] for row in items],
        )
        for name, items in sorted(groups.items())
    }


def confidence_interval_by_seed(rows):
    seed_scores = []
    for seed in sorted({row["seed"] for row in rows}):
        items = [row for row in rows if row["seed"] == seed]
        seed_scores.append(metrics([x["label"] for x in items], [x["prediction"] for x in items])["f1"])
    mean = statistics.mean(seed_scores)
    se = statistics.stdev(seed_scores) / math.sqrt(len(seed_scores)) if len(seed_scores) > 1 else 0.0
    return {"mean_f1": round(mean, 4), "ci95": [round(max(0.0, mean - 1.96 * se), 4), round(min(1.0, mean + 1.96 * se), 4)]}


def weight_sensitivity(rows):
    """Evaluate ranking only; weights never affect the triage decision."""
    schemes = {
        "equal": {"provenance": 1, "specificity": 1, "path_diversity": 1, "stability": 1, "closure_clearance": 1},
        "provenance_heavy": {"provenance": 4, "specificity": 1, "path_diversity": 2, "stability": 2, "closure_clearance": 1},
        "specificity_heavy": {"provenance": 2, "specificity": 4, "path_diversity": 1, "stability": 2, "closure_clearance": 1},
        "path_heavy": {"provenance": 2, "specificity": 1, "path_diversity": 4, "stability": 2, "closure_clearance": 1},
        "stability_heavy": {"provenance": 2, "specificity": 1, "path_diversity": 2, "stability": 4, "closure_clearance": 1},
    }
    labels = [row["label"] for row in rows]
    k = sum(labels)
    results = {}
    for name, weights in schemes.items():
        denominator = sum(weights.values())
        scores = [
            sum(weights[key] * row["metrics"][key] for key in weights) / denominator
            for row in rows
        ]
        results[name] = ranking_metrics(labels, scores, k)
    return results


def sensitivity_analysis(seed_start: int, repetitions: int):
    grid = []
    calibration_cases = {
        "supported_two_path",
        "single_path",
        "source_dependent_paths",
        "plausible_missing_edge",
        "stable_specific_community",
        "plausible_bridge_additions",
        "normalised_decline",
        "raw_decline_but_rate_stable",
    }
    for edge_keep in (0.70, 0.80, 0.90, 0.95):
        for repeats in (50, 100, 200):
            for threshold in (0.70, 0.80, 0.90, 0.95):
                rows = evaluate(
                    seed_start,
                    repetitions,
                    cfg_factory=lambda seed, ek=edge_keep, b=repeats, t=threshold: config(seed, ek, b, t),
                    case_filter=calibration_cases,
                )
                result = metrics([r["label"] for r in rows], [r["prediction"] for r in rows])
                grid.append({"edge_and_paper_keep": edge_keep, "repeats": repeats, "stability_threshold": threshold, **result})
    grid.sort(key=lambda row: (-row["f1"], -row["precision"], -row["recall"], row["repeats"]))
    return grid


def run(development_repetitions: int = 10, test_repetitions: int = 50):
    development = sensitivity_analysis(0, development_repetitions)
    selected = development[0]
    test_rows = evaluate(
        10_000,
        test_repetitions,
        cfg_factory=lambda seed: config(
            seed,
            selected["edge_and_paper_keep"],
            selected["repeats"],
            selected["stability_threshold"],
        ),
    )
    labels = [row["label"] for row in test_rows]
    full_predictions = [row["prediction"] for row in test_rows]
    ablation_names = [
        "provenance_only",
        "specificity_only",
        "path_diversity_only",
        "perturbation_only",
        "hard_contract_without_perturbation",
        "full_without_coverage_screen",
        "full_evidence_contract",
    ]
    ablations = {
        name: metrics(labels, [_prediction_for_ablation(name, row) for row in test_rows])
        for name in ablation_names
    }
    positives = sum(labels)
    ranking = {
        "detector_score": ranking_metrics(labels, [row["detector_score"] for row in test_rows], positives),
        "evidence_ranking_score": ranking_metrics(labels, [row["ranking_score"] for row in test_rows], positives),
    }
    return {
        "benchmark": "disjoint-development controlled multi-signal validation",
        "development_repetitions": development_repetitions,
        "test_repetitions": test_repetitions,
        "test_candidates": len(test_rows),
        "test_positives": positives,
        "binary_mapping": BINARY_MAPPING,
        "selected_on_development_only": {
            "edge_and_paper_keep": selected["edge_and_paper_keep"],
            "repeats": selected["repeats"],
            "stability_threshold": selected["stability_threshold"],
        },
        "raw_every_signal": metrics(labels, [1] * len(labels)),
        "full_evidence_contract": metrics(labels, full_predictions),
        "f1_across_test_seeds": confidence_interval_by_seed(test_rows),
        "by_signal_type": grouped_results(test_rows, "signal_type"),
        "by_case_type": grouped_results(test_rows, "case_type"),
        "ablations": ablations,
        "ranking": ranking,
        "ranking_weight_sensitivity": weight_sensitivity(test_rows),
        "development_sensitivity_grid": development,
        "scope": (
            "Controlled artefact rejection and ranking on disjoint synthetic seeds; "
            "not evidence of real-world novelty, usefulness, or expert acceptance."
        ),
        "decisions": test_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-repetitions", type=int, default=10)
    parser.add_argument("--test-repetitions", type=int, default=50)
    parser.add_argument("--output", default="outputs/validation_benchmark_v3.json")
    args = parser.parse_args()
    report = run(args.development_repetitions, args.test_repetitions)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"decisions", "development_sensitivity_grid"}}, indent=2))


if __name__ == "__main__":
    main()
