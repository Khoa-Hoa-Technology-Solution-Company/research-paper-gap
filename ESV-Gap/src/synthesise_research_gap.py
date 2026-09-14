"""Synthesize detector signals into reproducible, answerable research gaps.

The operational contract follows Robinson, Saldanha, and McKoy's two-part
framework: state where evidence is inadequate using PICOS-like elements and
state why the evidence is inadequate.  For computing research, PICOS is
adapted to Problem, Method, Comparator, Outcome, Setting/Dataset, and Timestamp
(PMCOST).  A candidate is accepted by predeclared hard gates; reviewer choices
do not participate in the decision.
"""

from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path
from typing import Any

import networkx as nx

from src.detect_gaps import limitation_evidence_semantics
from src.gap_certificate import build_gap_certificate
from src.utils import ensure_dir, get_logger, load_json, save_json
from src.validate_gaps import document_resolution_matches, load_corpus_documents


logger = get_logger("synthesise_gap")

FRAMEWORK_REFERENCES = [
    {
        "name": "Robinson research-gap framework",
        "citation": "Robinson KA, Saldanha IJ, McKoy NA. J Clin Epidemiol. 2011.",
        "doi": "10.1016/j.jclinepi.2011.06.009",
        "use": "Classify why evidence is inadequate and structure the gap with PICOS elements.",
    },
    {
        "name": "EPICOT+ research recommendation format",
        "citation": "Brown P et al. BMJ. 2006;333:804-806.",
        "doi": "10.1136/bmj.38987.492014.94",
        "use": "Record current evidence, answerable question elements, and timestamp.",
    },
]

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "using", "with", "without", "existing", "proposed",
}
_TOKEN_ALIASES = {
    "data": "dataset", "datasets": "dataset", "corpus": "dataset",
    "corpora": "dataset", "trained": "train", "training": "train",
    "methods": "method", "models": "model", "systems": "system",
    "expressions": "expression", "symbols": "symbol", "results": "result",
}
_CLOSING_RELATIONS = {"ADDRESSES", "IMPROVES", "PRODUCES", "EXTENDS"}
_GENERIC_STRUCTURAL_LABELS = {
    "accuracy", "approach", "architecture", "efficiency", "framework",
    "method", "model", "performance", "result", "solution", "system",
    "technique", "technology",
}


def concept_tokens(value: Any) -> set[str]:
    """Normalize light lexical variants for deterministic signal matching."""
    raw = re.findall(r"[a-z0-9]+", str(value or "").casefold().replace("-", " "))
    result = {
        _TOKEN_ALIASES.get(token, token)
        for token in raw
        if token not in _STOPWORDS and len(token) > 1
    }
    return result


def concept_similarity(left: Any, right: Any) -> float:
    """Coverage similarity; suitable for a short concept vs a longer label."""
    left_tokens = concept_tokens(left)
    right_tokens = concept_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / min(
        len(left_tokens), len(right_tokens)
    )


def _specific_structural_label(value: Any) -> bool:
    """Reject one-word umbrella labels as independent structural evidence."""
    tokens = concept_tokens(value)
    return bool(tokens) and not (
        len(tokens) == 1 and next(iter(tokens)) in _GENERIC_STRUCTURAL_LABELS
    )


def _paper_index(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {
        str(paper.get("paperId") or paper.get("paper_id") or paper.get("id")): paper
        for paper in documents
        if paper.get("paperId") or paper.get("paper_id") or paper.get("id")
    }
    return result


def limitation_semantic_audit(candidate: dict[str, Any]) -> dict[str, Any]:
    """Accept only limitation spans with safe polarity and attributable scope."""
    accepted = []
    rejected = []
    subject = candidate.get("subject", "")
    for record in candidate.get("source_evidence", []):
        semantics = limitation_evidence_semantics(
            record.get("evidence", ""),
            record.get("source_subject") or subject,
        )
        enriched = {**record, **semantics}
        (accepted if semantics["valid"] else rejected).append(enriched)
    scopes = {item.get("scope") for item in accepted}
    return {
        "passed": bool(accepted),
        "scope": "field" if "field" in scopes else ("subject" if accepted else None),
        "accepted_evidence": accepted,
        "rejected_evidence": rejected,
    }


def source_resolution_edges(
    G: nx.Graph,
    candidate: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    """Find evidence that the same paper/method already addresses the limitation."""
    subject = candidate.get("subject")
    missing = candidate.get("missing_capability", "")
    if not subject or not G.has_node(subject):
        return []
    supporting = {str(value) for value in candidate.get("supporting_paper_ids", [])}
    resolved = []
    outgoing = G.out_edges(subject, keys=True, data=True) if G.is_multigraph() else G.out_edges(subject, data=True)
    for edge in outgoing:
        target, data = edge[1], edge[-1]
        relation = str(data.get("relation", "")).upper()
        paper_id = str(data.get("source_paper") or data.get("source_paper_id") or "")
        if relation not in _CLOSING_RELATIONS or paper_id not in supporting:
            continue
        if concept_similarity(target, missing) < threshold:
            continue
        resolved.append({
            "subject": str(subject),
            "relation": relation,
            "object": str(target),
            "paper_id": paper_id,
            "evidence": data.get("evidence", ""),
        })
    return resolved


def source_resolution_documents(
    candidate: dict[str, Any],
    documents: list[dict[str, Any]],
    token_coverage: float,
) -> list[dict[str, Any]]:
    """Inspect complete source abstracts for solution clauses.

    Triple extraction can attach the proposed solution to a different subject
    node than the limitation. The source-paper text is therefore a mandatory,
    independent non-resolution gate rather than a fallback.
    """
    supporting = {
        str(value) for value in candidate.get("supporting_paper_ids", []) if value
    }
    resolved = []
    for paper in documents:
        paper_id = str(
            paper.get("paperId") or paper.get("paper_id") or paper.get("id") or ""
        )
        if paper_id not in supporting:
            continue
        matches = document_resolution_matches(
            candidate, paper, token_coverage=token_coverage
        )
        if not matches:
            continue
        resolved.append({
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "year": paper.get("year") or paper.get("publication_year"),
            "resolution_matches": matches,
        })
    return resolved


def convergent_signals(
    candidate: dict[str, Any],
    raw_gaps: dict[str, list[dict[str, Any]]],
    G: nx.Graph,
    documents: list[dict[str, Any]],
    similarity_threshold: float,
    max_direct_documents: int,
) -> dict[str, Any]:
    """Collect independent signal families supporting evidence inadequacy."""
    gap_type = candidate.get("type")
    if gap_type == "missing_link":
        subject = candidate.get("head", "")
        missing = candidate.get("tail", "")
    else:
        subject = candidate.get("subject", "")
        missing = candidate.get("missing_capability", "")
    scope_anchor = (
        candidate.get("domain", "")
        if candidate.get("semantic_scope") == "field"
        else subject
    )
    target = f"{scope_anchor} {missing}"
    if gap_type == "missing_link":
        families: dict[str, list[dict[str, Any]]] = {
            "missing_connection": [{
                "head": subject,
                "tail": missing,
                "detector": candidate.get("detector", "TransE"),
            }]
        }
        evidence_paths = candidate.get("validation", {}).get(
            "independent_evidence_paths", []
        )
        if evidence_paths:
            families["source_disjoint_evidence_paths"] = evidence_paths
    else:
        families = {"explicit_limitation": [{
            "subject": subject,
            "missing_capability": missing,
            "supporting_paper_ids": candidate.get("supporting_paper_ids", []),
        }]}

    if gap_type != "missing_link" and len(set(candidate.get("supporting_paper_ids", []))) >= 2:
        recurrence_name = (
            "evidence_cell_recurrence"
            if candidate.get("evidence_cell_id") else "independent_recurrence"
        )
        families[recurrence_name] = [{
            "supporting_paper_count": len(set(candidate.get("supporting_paper_ids", [])))
        }]

    source_papers = {
        str(value) for value in candidate.get("supporting_paper_ids", []) if value
    }
    orphan_matches = []
    for orphan in raw_gaps.get("orphan_clusters", []):
        matched_members = [
            str(member) for member in orphan.get("members", [])
            if _specific_structural_label(member) and max(
                concept_similarity(member, missing),
                concept_similarity(member, scope_anchor),
            ) >= similarity_threshold
        ]
        if matched_members:
            members = {str(value) for value in orphan.get("members", [])}
            cluster_papers: set[str] = set()
            graph_edges = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
            for edge in graph_edges:
                left, right, data = str(edge[0]), str(edge[1]), edge[-1]
                if left not in members or right not in members:
                    continue
                # A paper elsewhere in the same community does not corroborate
                # the matched concept. Require edge-level incidence.
                if left not in matched_members and right not in matched_members:
                    continue
                paper_id = data.get("source_paper") or data.get("source_paper_id") or data.get("paper_id")
                if paper_id:
                    cluster_papers.add(str(paper_id))
            independent_papers = sorted(cluster_papers.difference(source_papers))
            orphan_matches.append({
                "matched_members": matched_members,
                "inter_edge_ratio": orphan.get("inter_edge_ratio"),
                "cluster_size": orphan.get("cluster_size", len(orphan.get("members", []))),
                "supporting_paper_ids": sorted(cluster_papers),
                "source_disjoint_paper_ids": independent_papers,
            })
    independent_orphan_matches = [
        item for item in orphan_matches if item["source_disjoint_paper_ids"]
    ]
    if independent_orphan_matches:
        families["structural_undercoverage"] = independent_orphan_matches
    elif orphan_matches:
        families["same_source_structural_context"] = orphan_matches

    if gap_type != "missing_link":
        link_matches = []
        for link in raw_gaps.get("missing_links", []):
            link_text = f"{link.get('head', '')} {link.get('tail', '')}"
            if concept_similarity(link_text, target) >= similarity_threshold:
                link_matches.append({
                    "head": link.get("head"),
                    "tail": link.get("tail"),
                    "detector": link.get("detector", "TransE"),
                })
        if link_matches:
            families["missing_connection"] = link_matches

    temporal_matches = [
        item for item in raw_gaps.get("temporal_decay", [])
        if max(
            concept_similarity(item.get("concept", ""), scope_anchor),
            concept_similarity(item.get("concept", ""), missing),
        ) >= similarity_threshold
    ]
    if temporal_matches:
        families["temporal_undercoverage"] = [{
            "concept": item.get("concept"),
            "decay_rate": item.get("decay_rate"),
        } for item in temporal_matches]

    subject_tokens = concept_tokens(scope_anchor)
    missing_tokens = concept_tokens(missing)
    direct_documents = []
    for paper in documents:
        text = concept_tokens(f"{paper.get('title', '')} {paper.get('abstract', '')}")
        subject_match = bool(subject_tokens) and (
            len(subject_tokens.intersection(text)) / len(subject_tokens) >= 0.6
        )
        missing_match = bool(missing_tokens) and (
            len(missing_tokens.intersection(text)) / len(missing_tokens) >= 0.6
        )
        if subject_match and missing_match:
            direct_documents.append(str(
                paper.get("paperId") or paper.get("paper_id") or paper.get("id") or ""
            ))
    if len(direct_documents) <= max_direct_documents:
        families["corpus_scarcity"] = [{
            "direct_document_count": len(direct_documents),
            "paper_ids": [value for value in direct_documents if value],
        }]
    return families


def _nearby_typed_nodes(
    G: nx.Graph,
    seeds: list[str],
    node_type: str,
    limit: int = 3,
) -> list[str]:
    undirected = G.to_undirected()
    ranked = []
    for node, attrs in G.nodes(data=True):
        if str(attrs.get("type", "")).upper() != node_type:
            continue
        distances = []
        for seed in seeds:
            if seed and G.has_node(seed):
                try:
                    distances.append(nx.shortest_path_length(undirected, seed, node))
                except nx.NetworkXNoPath:
                    pass
        if distances and min(distances) <= 2:
            ranked.append((min(distances), -G.degree(node), str(node)))
    ranked.sort()
    return [item[2] for item in ranked[:limit]]


def _external_closure_gate(
    candidate: dict[str, Any],
    min_completed_queries: int,
) -> dict[str, Any]:
    validation = candidate.get("validation", {}) or {}
    record = validation.get("external_closure_search", {}) or {}
    completed = int(record.get(
        "completed_query_count",
        1 if record.get("performed") else 0,
    ))
    hits = record.get("closure_hits", []) or []
    return {
        "passed": bool(record.get("performed")) and completed >= min_completed_queries and not hits,
        "performed": bool(record.get("performed")),
        "completed_query_count": completed,
        "required_query_count": min_completed_queries,
        "closure_hits": hits,
        "record": record,
    }


def _operational_outcome_fallback(limitation: str) -> list[str]:
    """Map a stated limitation to measurable, non-placeholder outcomes."""
    text = str(limitation).casefold()
    if any(term in text for term in ("vulnerab", "xss", "attack", "exploit")):
        return ["confirmed vulnerability count", "exploit success rate"]
    if any(term in text for term in ("isolat", "contain", "component")):
        return ["cross-component compromise propagation rate", "runtime overhead"]
    if any(term in text for term in ("encrypt", "cryptograph")):
        return ["encryption throughput", "p95 operation latency", "CPU overhead"]
    if any(term in text for term in ("latency", "performance", "high-volume", "throughput")):
        return ["throughput", "p95 latency", "resource overhead"]
    if any(term in text for term in ("flexib", "maintain", "modif", "evolv")):
        return ["change lead time", "deployment coupling", "regression rate"]
    if any(term in text for term in ("accuracy", "recognition", "detect")):
        return ["task accuracy", "false-positive rate", "false-negative rate"]
    return ["capability success rate", "failure rate", "resource overhead"]


def _operational_setting_fallback(domain: str) -> list[str]:
    text = str(domain).casefold()
    if any(term in text for term in ("software", "monolith", "microservice", "security")):
        return [
            "representative open-source applications with reproducible workloads and security tests"
        ]
    return [f"a reproducible benchmark corpus representative of {domain}"]


def _researchable_specification(
    candidate: dict[str, Any],
    semantic_scope: str,
    G: nx.Graph,
    domain: str,
    snapshot_date: str,
) -> dict[str, Any]:
    gap_type = candidate.get("type")
    if gap_type == "missing_link":
        subject = str(candidate.get("head", "")).strip()
        limitation = str(candidate.get("tail", "")).strip()
    else:
        subject = str(candidate.get("subject", "")).strip()
        limitation = str(candidate.get("missing_capability", "")).strip()
    research_frame = candidate.get("research_frame", {}) or {}
    datasets = list(research_frame.get("datasets", [])) or _nearby_typed_nodes(
        G, [subject, limitation], "DATASET"
    )
    outcomes = list(research_frame.get("metrics", [])) or _nearby_typed_nodes(
        G, [subject, limitation], "METRIC"
    )
    if not datasets:
        datasets = _operational_setting_fallback(domain)
    if not outcomes:
        outcomes = _operational_outcome_fallback(limitation)
    if gap_type == "missing_link":
        subject_type = str(G.nodes[subject].get("type", "")) if G.has_node(subject) else ""
        limitation_type = str(G.nodes[limitation].get("type", "")) if G.has_node(limitation) else ""
        if subject_type.upper() == "METHOD":
            method = f"applying {subject} to {limitation}"
        elif limitation_type.upper() == "METHOD":
            method = f"applying {limitation} to {subject}"
        else:
            method = f"an approach integrating {subject} with {limitation}"
        comparator = "the closest independently supported baselines in the screened corpus"
    elif semantic_scope == "field":
        method = f"a method designed to address {limitation}"
        comparator = "current methods represented in the screened corpus"
    else:
        method = f"an enhanced {subject} that addresses {limitation}"
        comparator = f"the reported {subject} baseline"
    setting = ", ".join(datasets)
    outcome = ", ".join(outcomes)
    question = (
        f"In {domain}, does {method}, compared with {comparator}, improve {outcome} "
        f"when evaluated on {setting}?"
    )
    if gap_type == "missing_link":
        claim = (
            f"As of {snapshot_date}, the screened evidence is insufficient to "
            f"determine the direct relationship between '{subject}' and "
            f"'{limitation}' in {domain}."
        )
    else:
        claim = (
            f"As of {snapshot_date}, the screened evidence is insufficient to determine "
            f"how to address '{limitation}' in {domain}."
        )
    return {
        "claim": claim,
        "research_question": question,
        "PMCOST": {
            "problem": domain,
            "method_or_intervention": method,
            "comparator": comparator,
            "outcomes": outcomes,
            "setting_or_datasets": datasets,
            "timestamp": snapshot_date,
        },
        "suggested_study": {
            "design": "controlled benchmark comparison",
            "intervention": method,
            "comparator": comparator,
            "datasets": datasets,
            "primary_outcomes": outcomes,
            "minimum_reporting": [
                "predefined train/validation/test split",
                "same-data and same-compute comparison",
                "uncertainty intervals or repeated-run variance",
                "error analysis for the stated limitation",
            ],
        },
        "answerable": bool(domain and subject and limitation and datasets and outcomes),
    }


def evaluate_synthesis_candidate(
    candidate: dict[str, Any],
    raw_gaps: dict[str, list[dict[str, Any]]],
    G: nx.Graph,
    documents: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared automatic research-gap synthesis contract."""
    settings = config.get("gap_synthesis", {})
    validation = candidate.get("validation", {}) or {}
    gap_type = candidate.get("type")
    similarity_threshold = float(settings.get("signal_similarity_threshold", 0.60))
    if gap_type == "missing_link":
        required_paths = int(config.get("gap_validation", {}).get("min_independent_paths", 2))
        path_count = int(validation.get("independent_evidence_path_count", 0))
        semantic = {
            "passed": path_count >= required_paths,
            "scope": "evidence_map_cell",
            "accepted_evidence": [],
            "rejected_evidence": [],
            "reason": "source_disjoint_empty_evidence_cell",
            "independent_evidence_path_count": path_count,
            "required_path_count": required_paths,
        }
    else:
        semantic = limitation_semantic_audit(candidate)
    candidate_for_contract = {
        **candidate,
        "semantic_scope": semantic.get("scope") or candidate.get("semantic_scope"),
        "domain": candidate.get("domain") or config.get("project", {}).get("domain", ""),
    }
    resolution_edges = (
        [] if gap_type == "missing_link"
        else source_resolution_edges(G, candidate_for_contract, similarity_threshold)
    )
    resolution_documents = (
        [] if gap_type == "missing_link"
        else source_resolution_documents(
            candidate_for_contract,
            documents,
            float(config.get("gap_validation", {}).get(
                "closure_token_coverage", 0.60
            )),
        )
    )
    signals = convergent_signals(
        candidate_for_contract,
        raw_gaps,
        G,
        documents,
        similarity_threshold,
        int(settings.get("max_direct_documents", 2)),
    )
    graph_families = {
        name for name in signals
        if name in {
            "structural_undercoverage", "missing_connection",
            "source_disjoint_evidence_paths", "temporal_undercoverage",
            "evidence_cell_recurrence",
        }
    }
    evidentiary_signals = {
        name: value for name, value in signals.items()
        if name not in {"corpus_scarcity", "same_source_structural_context"}
    }
    closure = _external_closure_gate(
        candidate_for_contract,
        int(settings.get("min_external_queries", 2)),
    )
    formulation = _researchable_specification(
        candidate_for_contract,
        semantic.get("scope") or "subject",
        G,
        str(config.get("project", {}).get("domain", "")).strip(),
        str(config.get("gap_validation", {}).get("snapshot_date", "")),
    )
    corpus_minimum = int(settings.get("min_screened_corpus_size", 30))
    validation_settings = config.get("gap_validation", {})
    stability = float(validation.get("metrics", {}).get("stability", 0.0))
    specificity = float(validation.get("metrics", {}).get("specificity", 0.0))
    recorded_problem_relevance = validation.get("metrics", {}).get(
        "candidate_problem_relevance"
    )
    minimum_problem_relevance = float(
        validation_settings.get("min_problem_relevance", 0.40)
    )
    problem_alignment_passed = bool(validation.get("problem_relevant", True))
    if recorded_problem_relevance is not None:
        problem_alignment_passed = (
            problem_alignment_passed
            and float(recorded_problem_relevance) >= minimum_problem_relevance
        )
    hard_gates = {
        "limitation_or_empty_cell_supported": semantic["passed"],
        "candidate_domain_relevant": validation.get("domain_relevant", True),
        "candidate_problem_aligned_with_domain": problem_alignment_passed,
        "not_already_resolved": (
            not resolution_edges
            and not resolution_documents
            and not validation.get("existing_direct_edge", False)
        ),
        "minimum_screened_corpus": len(documents) >= corpus_minimum,
        "stable_under_perturbation": stability >= float(
            validation_settings.get("min_stability", 0.70)
        ),
        "specific_question_entities": specificity >= float(
            validation_settings.get("min_specificity", 0.55)
        ),
        "local_closure_clear": (
            bool(validation.get("closure_corpus_available"))
            and not validation.get("closure_hits", [])
        ),
        "external_closure_clear": closure["passed"],
        "multi_signal_convergence": (
            len(evidentiary_signals) >= int(settings.get("min_signal_families", 2))
            and len(graph_families) >= int(settings.get("min_graph_signal_families", 1))
        ),
        "answerable_research_question": formulation["answerable"],
    }
    passed = all(hard_gates.values())
    confidence = (
        0.25 * float(
            candidate.get("mean_evidence_confidence")
            if candidate.get("mean_evidence_confidence") is not None
            else min(abs(float(candidate.get("prediction_score", 0.0))) / 10.0, 1.0)
        )
        + 0.20 * min(
            len(set(
                candidate.get("supporting_paper_ids", [])
                or validation.get("supporting_paper_ids", [])
            )) / 2,
            1.0,
        )
        + 0.20 * min(len(evidentiary_signals) / 3, 1.0)
        + 0.15 * stability
        + 0.20 * float(closure["passed"])
    )
    identity_material = "|".join([
        str(candidate.get("subject") or candidate.get("head") or ""),
        str(candidate.get("missing_capability") or candidate.get("tail") or ""),
        str(config.get("gap_validation", {}).get("snapshot_date", "")),
    ])
    result = {
        "gap_id": "srg_" + hashlib.sha256(identity_material.encode("utf-8")).hexdigest()[:16],
        "status": (
            "automatically_supported_research_gap"
            if passed else "signal_bundle_rejected"
        ),
        "gap_reason": "insufficient_or_imprecise_information",
        **formulation,
        "decision": {
            "passed": passed,
            "hard_gates": hard_gates,
            "signal_family_count": len(evidentiary_signals),
            "contextual_signal_count": len(signals) - len(evidentiary_signals),
            "graph_signal_family_count": len(graph_families),
            "automatic_strength_score": round(confidence, 4),
            "rule": "all hard gates must pass; score is used only to choose among passing gaps",
        },
        "semantic_audit": semantic,
        "source_resolution_edges": resolution_edges,
        "source_resolution_documents": resolution_documents,
        "convergent_signals": signals,
        "external_closure": closure,
        "supporting_paper_ids": sorted(
            {
                str(item.get("paper_id"))
                for item in semantic.get("accepted_evidence", [])
                if item.get("paper_id")
            }
            or {
                str(paper_id)
                for paper_id in validation.get("supporting_paper_ids", [])
                if paper_id
            }
        ),
        "source_candidate": candidate,
    }
    certificate = build_gap_certificate(result, documents, config)
    result["gap_certificate"] = certificate
    if certificate["passed"]:
        result["status"] = "certified_corpus_bounded_research_gap"
    elif result["decision"]["passed"] and certificate.get("enabled"):
        result["status"] = "evidence_cleared_hypothesis_not_certified"
    return result


def synthesise_research_gap(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate explicit limitations and evidence-map empty cells; emit one gap."""
    output_dir = ensure_dir(config["paths"]["outputs"])
    graph_path = Path(config["paths"]["graph"]) / "knowledge_graph.pkl"
    raw_path = Path(output_dir) / "detected_gaps_raw.json"
    audit_path = Path(output_dir) / "gap_validation_audit.json"
    if not graph_path.exists() or not raw_path.exists() or not audit_path.exists():
        raise FileNotFoundError("Run build, detect, and validate before gap synthesis")

    with open(graph_path, "rb") as stream:
        G = pickle.load(stream)
    raw_gaps = load_json(raw_path)
    validation_audit = load_json(audit_path)
    documents = list(load_corpus_documents(config) or [])
    paper_index = _paper_index(documents)

    evaluated = []
    for candidate in validation_audit.get("candidates", []):
        if candidate.get("type") not in {"evidence_gap", "missing_link"}:
            continue
        result = evaluate_synthesis_candidate(candidate, raw_gaps, G, documents, config)
        corroborating_ids = sorted({
            str(paper_id)
            for records in result.get("convergent_signals", {}).values()
            for record in (records if isinstance(records, list) else [])
            if isinstance(record, dict)
            for paper_id in record.get("source_disjoint_paper_ids", [])
            if paper_id
        })
        result["corroborating_paper_ids"] = corroborating_ids
        result["source_papers"] = [
            {
                "paper_id": paper_id,
                "title": paper_index.get(paper_id, {}).get("title", ""),
                "year": paper_index.get(paper_id, {}).get("year"),
                "url": paper_index.get(paper_id, {}).get("url", ""),
            }
            for paper_id in result["supporting_paper_ids"]
        ]
        result["corroborating_papers"] = [
            {
                "paper_id": paper_id,
                "title": paper_index.get(paper_id, {}).get("title", ""),
                "year": paper_index.get(paper_id, {}).get("year"),
                "url": paper_index.get(paper_id, {}).get("url", ""),
            }
            for paper_id in corroborating_ids
        ]
        evaluated.append(result)

    qualified = [item for item in evaluated if item.get("gap_certificate", {}).get("passed")]
    qualified.sort(
        key=lambda item: (
            item["decision"]["automatic_strength_score"],
            item["decision"]["signal_family_count"],
            item["gap_id"],
        ),
        reverse=True,
    )
    primary = qualified[0] if qualified else None
    rejection_summary: dict[str, int] = {}
    for item in evaluated:
        if item["decision"]["passed"]:
            continue
        for gate, passed_gate in item["decision"]["hard_gates"].items():
            if not passed_gate:
                rejection_summary[gate] = rejection_summary.get(gate, 0) + 1
    rejection_summary = dict(sorted(
        rejection_summary.items(), key=lambda pair: (-pair[1], pair[0])
    ))
    certificate_rejection_counts: dict[str, int] = {}
    for item in evaluated:
        certificate = item.get("gap_certificate", {}) or {}
        for gate in certificate.get("failed_gates", []):
            certificate_rejection_counts[gate] = (
                certificate_rejection_counts.get(gate, 0) + 1
            )
    certificate_rejection_counts = dict(sorted(
        certificate_rejection_counts.items(), key=lambda pair: (-pair[1], pair[0])
    ))
    next_actions = []
    if rejection_summary.get("multi_signal_convergence"):
        next_actions.append(
            "Expand the screened corpus with focused limitation/future-work queries; "
            "same-source topology and corpus scarcity do not count as convergence."
        )
    if rejection_summary.get("candidate_problem_aligned_with_domain"):
        next_actions.append(
            "Refine search queries around the domain's problem outcomes; candidates "
            "about unrelated performance or flexibility are intentionally rejected."
        )
    closure_is_actual_blocker = any(
        not item["decision"]["hard_gates"].get("external_closure_clear", False)
        and all(item["decision"]["hard_gates"].get(gate, False) for gate in (
            "limitation_or_empty_cell_supported",
            "candidate_domain_relevant",
            "candidate_problem_aligned_with_domain",
            "multi_signal_convergence",
        ))
        for item in evaluated
    )
    if closure_is_actual_blocker:
        next_actions.append(
            "Complete or retry candidate-specific external closure searches before "
            "making an automatic gap claim."
        )
    if certificate_rejection_counts.get("source_text_evidence_sufficient"):
        next_actions.append(
            "Acquire and parse full text for every limitation source; abstracts "
            "are sufficient for a hypothesis but never for a certified gap."
        )
    if certificate_rejection_counts.get("independent_limitation_sources"):
        next_actions.append(
            "Expand the corpus until at least two source-disjoint papers report "
            "the same limitation, or retain the result as a single-source hypothesis."
        )
    if certificate_rejection_counts.get("external_counterevidence_search_complete"):
        next_actions.append(
            "Retry every documented external counterevidence query; incomplete "
            "retrieval cannot support a certified corpus-bounded gap."
        )
    target_size = int(config.get("filtering", {}).get(
        "target_corpus_size", len(documents)
    ))
    if len(documents) < target_size:
        next_actions.append(
            f"Only {len(documents)} papers were retained from the target of "
            f"{target_size}; collect a larger query-balanced pool and rerun screening."
        )
    report = {
        "schema_version": 3,
        "operational_definition": (
            "A certified corpus-bounded research gap is an answerable question for "
            "which the synthesis contract passes, at least two source-disjoint "
            "limitation sources have sufficient source text, a frozen corpus "
            "fingerprint is recorded, and completed local and external "
            "counterevidence searches find no evidence that closes the limitation."
        ),
        "human_decision_used": False,
        "framework_references": FRAMEWORK_REFERENCES,
        "corpus": {
            "screened_paper_count": len(documents),
            "year_range": config.get("collection", {}).get("year_range", []),
            "queries": config.get("collection", {}).get("queries", []),
            "snapshot_date": config.get("gap_validation", {}).get("snapshot_date"),
        },
        "evaluated_candidate_count": len(evaluated),
        "qualified_gap_count": len(qualified),
        "certified_gap_count": len(qualified),
        "primary_gap": primary,
        "qualified_gaps": qualified,
        "certified_gaps": qualified,
        "candidate_audit": evaluated,
        "rejection_gate_counts": rejection_summary,
        "certificate_rejection_gate_counts": certificate_rejection_counts,
        "next_actions": next_actions,
        "null_result_note": (
            None if primary else
            "No candidate satisfied every certification gate for a corpus-bounded "
            "research gap. This is not evidence that no global research gap exists; "
            "it means the recorded corpus, source text, or closure-search evidence "
            "is insufficient for that conclusion."
        ),
    }
    save_json(report, Path(output_dir) / "research_gap_synthesis.json")
    save_json(
        {
            "schema_version": 1,
            "certified_gap_count": len(qualified),
            "certificates": [item["gap_certificate"] for item in evaluated],
        },
        Path(output_dir) / "gap_certificates.json",
    )
    save_json(
        primary or {
            "status": "no_certified_corpus_bounded_research_gap",
            "reason": report["null_result_note"],
        },
        Path(output_dir) / "primary_research_gap.json",
    )
    logger.info(
        "Research-gap certification: %d evaluated, %d certified, primary=%s",
        len(evaluated),
        len(qualified),
        primary.get("gap_id") if primary else "none",
    )
    return report


if __name__ == "__main__":
    import yaml

    with open("config.yaml", encoding="utf-8") as stream:
        synthesise_research_gap(yaml.safe_load(stream))
