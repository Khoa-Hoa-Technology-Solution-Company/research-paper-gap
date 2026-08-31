"""Leakage-controlled temporal backtesting for research-gap candidates.

The backtest deliberately uses an outcome-based *silver standard*.  A paper
published after cutoff ``t`` is a positive control only when its extracted
record both states a limitation and contains a solution/evaluation action for
that limitation, or when it instantiates a typed evidence-map relation that was
absent before ``t``.  A pre-cutoff limitation resolved in its own source is a
negative control.  Everything else remains unlabelled instead of being treated
as a false positive in an open literature world.

These labels measure anticipatory candidate utility.  They are not proof that
the post-cutoff paper was globally first, nor are they a substitute for expert
assessment of scientific novelty.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from src.detect_gaps import detect_evidence_gaps, detect_evidence_map_empty_cells


ACTION_RELATIONS = {
    "ADDRESSES", "IMPROVES", "PRODUCES", "EXTENDS", "APPLIED_TO",
    "EVALUATES_ON", "PROPOSES",
}
STRUCTURAL_RELATIONS = {"ADDRESSES", "APPLIED_TO", "EVALUATES_ON", "USES"}
ALLOWED_TYPE_PAIRS = {
    frozenset(("METHOD", "DATASET")),
    frozenset(("METHOD", "CONCEPT")),
    frozenset(("CONCEPT", "DATASET")),
}
SOLUTION_CUES = {
    "address", "addresses", "addressed", "overcome", "overcomes",
    "mitigate", "mitigates", "solve", "solves", "propose", "proposes",
    "present", "presents", "develop", "develops", "evaluate", "evaluates",
    "improve", "improves", "framework", "approach", "method", "model",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "using", "with", "without", "lack",
    "lacks", "limited", "limitation", "limitations", "insufficient",
}


def paper_id(document: dict[str, Any]) -> str:
    return str(document.get("paperId") or document.get("paper_id") or "")


def tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value).casefold())
        if len(token) > 1 and token not in STOPWORDS
    }


def token_similarity(left: Any, right: Any) -> float:
    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens.intersection(right_tokens))
    containment = overlap / min(len(left_tokens), len(right_tokens))
    jaccard = overlap / len(left_tokens.union(right_tokens))
    return 0.7 * containment + 0.3 * jaccard


def build_temporal_graph(triples: Iterable[dict[str, Any]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for triple in triples:
        subject = str(triple.get("subject", {}).get("name", "")).strip()
        obj = str(triple.get("object", {}).get("name", "")).strip()
        if not subject or not obj or subject == obj:
            continue
        source = str(triple.get("source_paper_id") or "")
        for label, entity in ((subject, triple.get("subject", {})), (obj, triple.get("object", {}))):
            if not graph.has_node(label):
                graph.add_node(
                    label,
                    type=str(entity.get("type") or "UNKNOWN").upper(),
                    papers=set(),
                )
            graph.nodes[label]["papers"].add(source)
        graph.add_edge(
            subject,
            obj,
            relation=str(triple.get("relation") or "RELATED").upper(),
            confidence=float(triple.get("confidence") or 0.0),
            source_paper=source,
            year=triple.get("source_year"),
            evidence=str(triple.get("evidence") or ""),
        )
    for node in graph.nodes:
        graph.nodes[node]["papers"] = sorted(
            value for value in graph.nodes[node]["papers"] if value
        )
    return graph


def _triple_text(triple: dict[str, Any]) -> str:
    return " ".join((
        str(triple.get("subject", {}).get("name", "")),
        str(triple.get("object", {}).get("name", "")),
        str(triple.get("evidence", "")),
    ))


def _solution_action(limitation: dict[str, Any], action: dict[str, Any]) -> bool:
    if str(action.get("relation", "")).upper() not in ACTION_RELATIONS:
        return False
    action_text = _triple_text(action)
    if not tokens(action_text).intersection(SOLUTION_CUES):
        return False
    limitation_object = limitation.get("object", {}).get("name", "")
    # Subject overlap alone is not resolution evidence.  For example, a paper
    # can discuss an IDS's explainability limitation while improving only its
    # attack-detection resilience.  The action must overlap the capability.
    return token_similarity(limitation_object, action_text) >= 0.28


def limitation_resolution_controls(
    triples: Iterable[dict[str, Any]],
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    label: str,
) -> list[dict[str, Any]]:
    """Return document-level limitation-plus-action controls."""
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for triple in triples:
        try:
            year = int(triple.get("source_year"))
        except (TypeError, ValueError):
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        by_paper[str(triple.get("source_paper_id") or "")].append(triple)

    controls: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source, records in by_paper.items():
        limitations = [
            record for record in records
            if str(record.get("relation", "")).upper() == "LACKS"
        ]
        actions = [
            record for record in records
            if str(record.get("relation", "")).upper() in ACTION_RELATIONS
        ]
        for limitation in limitations:
            matching_actions = [
                action for action in actions if _solution_action(limitation, action)
            ]
            if not matching_actions:
                continue
            capability = str(limitation.get("object", {}).get("name", "")).strip()
            key = (source, " ".join(sorted(tokens(capability))))
            if not capability or key in seen:
                continue
            seen.add(key)
            year = int(limitation.get("source_year"))
            identity = f"{label}|{source}|{key[1]}"
            controls.append({
                "control_id": "tc_" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                "control_type": "limitation_resolution",
                "label": label,
                "subject": str(limitation.get("subject", {}).get("name", "")),
                "missing_capability": capability,
                "paper_id": source,
                "year": year,
                "limitation_evidence": str(limitation.get("evidence") or ""),
                "resolution_relation": str(matching_actions[0].get("relation") or ""),
                "resolution_evidence": str(matching_actions[0].get("evidence") or ""),
            })
    return controls


def structural_future_controls(
    pre_graph: nx.MultiDiGraph,
    future_triples: Iterable[dict[str, Any]],
    minimum_marginal_papers: int = 2,
) -> list[dict[str, Any]]:
    """Find post-cutoff typed relations between independently known endpoints."""
    controls: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for triple in future_triples:
        relation = str(triple.get("relation") or "").upper()
        if relation not in STRUCTURAL_RELATIONS:
            continue
        head = str(triple.get("subject", {}).get("name", "")).strip()
        tail = str(triple.get("object", {}).get("name", "")).strip()
        if not pre_graph.has_node(head) or not pre_graph.has_node(tail):
            continue
        type_pair = frozenset((
            str(pre_graph.nodes[head].get("type", "UNKNOWN")).upper(),
            str(pre_graph.nodes[tail].get("type", "UNKNOWN")).upper(),
        ))
        if type_pair not in ALLOWED_TYPE_PAIRS:
            continue
        if pre_graph.has_edge(head, tail) or pre_graph.has_edge(tail, head):
            continue
        head_papers = set(pre_graph.nodes[head].get("papers", []))
        tail_papers = set(pre_graph.nodes[tail].get("papers", []))
        if min(len(head_papers), len(tail_papers)) < minimum_marginal_papers:
            continue
        key = tuple(sorted((head, tail))) + (relation,)
        if key in seen:
            continue
        seen.add(key)
        identity = "|".join(key)
        controls.append({
            "control_id": "tc_" + hashlib.sha256(identity.encode()).hexdigest()[:16],
            "control_type": "future_typed_relation",
            "label": "positive",
            "head": head,
            "tail": tail,
            "relation": relation,
            "paper_id": str(triple.get("source_paper_id") or ""),
            "year": int(triple.get("source_year")),
            "resolution_evidence": str(triple.get("evidence") or ""),
        })
    return controls


def candidate_score(candidate: dict[str, Any]) -> float:
    quality = candidate.get("candidate_quality", {}) or {}
    if quality.get("score") is not None:
        return float(quality["score"])
    return float(candidate.get("mean_evidence_confidence") or candidate.get("prediction_score") or 0.0)


def candidate_matches_control(candidate: dict[str, Any], control: dict[str, Any]) -> bool:
    if candidate.get("type") == "missing_link" and control.get("control_type") == "future_typed_relation":
        candidate_pair = {
            " ".join(sorted(tokens(candidate.get("head", "")))),
            " ".join(sorted(tokens(candidate.get("tail", "")))),
        }
        control_pair = {
            " ".join(sorted(tokens(control.get("head", "")))),
            " ".join(sorted(tokens(control.get("tail", "")))),
        }
        return candidate_pair == control_pair
    if candidate.get("type") != "evidence_gap" or control.get("control_type") != "limitation_resolution":
        return False
    capability_match = token_similarity(
        candidate.get("missing_capability", ""), control.get("missing_capability", "")
    )
    subject_match = token_similarity(
        candidate.get("subject", ""), control.get("subject", "")
    )
    return capability_match >= 0.62 and (subject_match >= 0.25 or capability_match >= 0.82)


def evaluate_ranked_candidates(
    candidates: list[dict[str, Any]],
    positive_controls: list[dict[str, Any]],
    negative_controls: list[dict[str, Any]],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, Any]:
    ranked = sorted(candidates, key=candidate_score, reverse=True)
    positive_matches = [
        {control["control_id"] for control in positive_controls if candidate_matches_control(candidate, control)}
        for candidate in ranked
    ]
    negative_matches = [
        {control["control_id"] for control in negative_controls if candidate_matches_control(candidate, control)}
        for candidate in ranked
    ]
    recall_at_k: dict[str, float | None] = {}
    precision_at_k: dict[str, float | None] = {}
    for k in ks:
        selected_positive = set().union(*positive_matches[:k]) if ranked[:k] else set()
        labelled_tp = sum(bool(value) for value in positive_matches[:k])
        labelled_fp = sum(bool(value) for value in negative_matches[:k])
        recall_at_k[str(k)] = (
            round(len(selected_positive) / len(positive_controls), 4)
            if positive_controls else None
        )
        precision_at_k[str(k)] = (
            round(labelled_tp / (labelled_tp + labelled_fp), 4)
            if labelled_tp + labelled_fp else None
        )

    # Local closure blocks candidates matching known pre-cutoff resolutions.
    closure_indices = [index for index, values in enumerate(negative_matches) if not values]
    closure_tp = sum(bool(positive_matches[index]) for index in closure_indices)
    closure_fp = sum(bool(negative_matches[index]) for index in closure_indices)
    precision_after_closure = (
        round(closure_tp / (closure_tp + closure_fp), 4)
        if closure_tp + closure_fp else None
    )
    selected_negative_controls = set().union(
        *(negative_matches[index] for index in closure_indices)
    ) if closure_indices else set()
    false_positive_rate = (
        round(len(selected_negative_controls) / len(negative_controls), 4)
        if negative_controls else None
    )
    unlabelled = sum(
        not positive_matches[index] and not negative_matches[index]
        for index in range(len(ranked))
    )
    return {
        "candidate_count": len(ranked),
        "positive_control_count": len(positive_controls),
        "negative_control_count": len(negative_controls),
        "candidate_recall_at_k": recall_at_k,
        "candidate_precision_at_k_on_labelled_controls": precision_at_k,
        "precision_after_local_closure_on_labelled_controls": precision_after_closure,
        "labelled_true_positives_after_local_closure": closure_tp,
        "labelled_false_positives_after_local_closure": closure_fp,
        "false_positive_rate_on_known_negatives": false_positive_rate,
        "known_negatives_selected_after_local_closure": len(selected_negative_controls),
        "unlabelled_candidate_count": unlabelled,
        "label_abstention_rate": round(unlabelled / len(ranked), 4) if ranked else None,
        # Certification is intentionally not replayed without historical full
        # text and historical external-search snapshots.
        "certified_candidate_count": 0,
        "certificate_precision": None,
        "certification_abstention_rate": 1.0 if ranked else None,
        "certificate_metric_status": "not_estimable_without_historical_full_text_and_closure_snapshots",
    }


def _write_jsonl(path: Path, documents: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for document in documents:
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")


def run_cutoff_backtest(
    documents: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    base_config: dict[str, Any],
    cutoff: int,
    output_dir: Path,
) -> dict[str, Any]:
    pre_documents = [document for document in documents if int(document.get("year") or 0) <= cutoff]
    future_documents = [document for document in documents if int(document.get("year") or 0) > cutoff]
    pre_ids = {paper_id(document) for document in pre_documents}
    future_ids = {paper_id(document) for document in future_documents}
    pre_triples = [triple for triple in triples if str(triple.get("source_paper_id") or "") in pre_ids]
    future_triples = [triple for triple in triples if str(triple.get("source_paper_id") or "") in future_ids]
    graph = build_temporal_graph(pre_triples)

    cutoff_dir = output_dir / f"cutoff_{cutoff}"
    processed_dir = cutoff_dir / "data" / "processed"
    graph_dir = cutoff_dir / "data" / "graph"
    outputs_dir = cutoff_dir / "outputs"
    _write_jsonl(processed_dir / "corpus_filtered.jsonl", pre_documents)
    graph_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    config = copy.deepcopy(base_config)
    config.setdefault("gap_validation", {})["snapshot_date"] = f"{cutoff}-12-31"
    config["paths"] = {
        "processed_data": str(processed_dir),
        "graph": str(graph_dir),
        "outputs": str(outputs_dir),
    }
    candidates = detect_evidence_gaps(graph, config) + detect_evidence_map_empty_cells(graph, config)
    candidates = sorted(candidates, key=candidate_score, reverse=True)

    positive_controls = limitation_resolution_controls(
        future_triples, start_year=cutoff + 1, label="positive"
    )
    positive_controls.extend(structural_future_controls(
        graph,
        future_triples,
        int(config.get("gap_detection", {}).get("evidence_map", {}).get("min_marginal_papers", 2)),
    ))
    negative_controls = limitation_resolution_controls(
        pre_triples, end_year=cutoff, label="negative"
    )
    metrics = evaluate_ranked_candidates(candidates, positive_controls, negative_controls)

    leakage_violations = []
    pre_year_by_id = {paper_id(document): int(document.get("year") or 0) for document in pre_documents}
    for index, candidate in enumerate(candidates):
        for source in candidate.get("supporting_paper_ids", []) or []:
            if source not in pre_year_by_id or pre_year_by_id[source] > cutoff:
                leakage_violations.append({"candidate_index": index, "paper_id": source})

    result = {
        "cutoff_year": cutoff,
        "prediction_window": [min((int(d.get("year") or 0) for d in future_documents), default=None),
                              max((int(d.get("year") or 0) for d in future_documents), default=None)],
        "pre_cutoff_papers": len(pre_documents),
        "post_cutoff_papers": len(future_documents),
        "pre_cutoff_triples": len(pre_triples),
        "post_cutoff_triples": len(future_triples),
        "leakage_check_passed": not leakage_violations,
        "leakage_violations": leakage_violations,
        "metrics": metrics,
        "positive_controls": positive_controls,
        "negative_controls": negative_controls,
        "ranked_candidates": candidates,
    }
    (cutoff_dir / "temporal_backtest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    def weighted(metric_name: str) -> float | None:
        values = []
        for result in results:
            value = result["metrics"].get(metric_name)
            if value is not None:
                values.append(float(value))
        return round(sum(values) / len(values), 4) if values else None

    recall_keys = sorted({
        key for result in results
        for key in result["metrics"]["candidate_recall_at_k"]
    }, key=int)
    macro_recall = {}
    for key in recall_keys:
        values = [
            result["metrics"]["candidate_recall_at_k"].get(key)
            for result in results
            if result["metrics"]["candidate_recall_at_k"].get(key) is not None
        ]
        macro_recall[key] = round(sum(values) / len(values), 4) if values else None
    total_closure_tp = sum(
        result["metrics"]["labelled_true_positives_after_local_closure"]
        for result in results
    )
    total_closure_fp = sum(
        result["metrics"]["labelled_false_positives_after_local_closure"]
        for result in results
    )
    micro_closure_precision = (
        round(total_closure_tp / (total_closure_tp + total_closure_fp), 4)
        if total_closure_tp + total_closure_fp else None
    )
    total_known_negative_selected = sum(
        result["metrics"]["known_negatives_selected_after_local_closure"]
        for result in results
    )
    total_negative_controls = sum(
        result["metrics"]["negative_control_count"] for result in results
    )
    return {
        "cutoff_count": len(results),
        "cutoffs": [result["cutoff_year"] for result in results],
        "all_leakage_checks_passed": all(result["leakage_check_passed"] for result in results),
        "total_candidates": sum(result["metrics"]["candidate_count"] for result in results),
        "total_positive_controls": sum(result["metrics"]["positive_control_count"] for result in results),
        "total_negative_controls": total_negative_controls,
        "macro_candidate_recall_at_k": macro_recall,
        "macro_precision_after_local_closure_on_labelled_controls": weighted(
            "precision_after_local_closure_on_labelled_controls"
        ),
        "micro_precision_after_local_closure_on_labelled_controls": micro_closure_precision,
        "labelled_true_positives_after_local_closure": total_closure_tp,
        "labelled_false_positives_after_local_closure": total_closure_fp,
        "macro_false_positive_rate_on_known_negatives": weighted(
            "false_positive_rate_on_known_negatives"
        ),
        "known_negatives_selected_after_local_closure": total_known_negative_selected,
        "macro_label_abstention_rate": weighted("label_abstention_rate"),
        "certificate_precision": None,
        "certification_abstention_rate": 1.0 if any(
            result["metrics"]["candidate_count"] for result in results
        ) else None,
    }
