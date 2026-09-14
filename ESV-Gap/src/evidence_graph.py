"""Paper-centred evidence graph for high-yield research-gap candidates.

The entity knowledge graph preserves extracted facts, but its topology is not
the right unit for deciding whether a research question is under-supported.
This module projects explicit limitations into evidence cells and connects
each cell to the independent papers and research-frame entities that support
it. Candidate ranking is therefore driven by evidence recurrence and study
context rather than by arbitrary proximity between extracted phrases.
"""

from __future__ import annotations

import hashlib
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from src.utils import ensure_dir, load_jsonl, save_json


_GENERIC = {
    "ability", "approach", "approaches", "capability", "challenge",
    "challenges", "current", "existing", "framework", "frameworks",
    "lack", "lacks", "limitation", "limitations", "method", "methods",
    "model", "models", "need", "needs", "proposed", "requirement",
    "requirements", "research", "system", "systems", "technique",
    "techniques", "the", "their", "use", "using", "with", "without",
    "and", "for", "under", "such", "across", "various", "which", "that",
}
_ALIASES = {
    "explainable": "explainability",
    "interpretable": "explainability",
    "interpretability": "explainability",
    "transparent": "explainability",
    "transparency": "explainability",
    "adversarially": "adversarial",
    "robust": "robustness",
    "datasets": "dataset",
    "data": "dataset",
    "benchmarks": "benchmark",
    "standardized": "standardization",
    "standardised": "standardization",
    "standardizing": "standardization",
    "computational": "compute",
    "computation": "compute",
    "complexity": "overhead",
    "efficient": "efficiency",
    "lightweight": "efficiency",
    "realistic": "real_world",
    "reality": "real_world",
    "generalisation": "generalization",
    "generalize": "generalization",
    "generalise": "generalization",
    "privacy-preserving": "privacy",
}
_PHRASES = {
    r"\breal[- ]world\b": "real_world",
    r"\bzero[- ]day\b": "zero_day",
    r"\bcross[- ]dataset\b": "cross_dataset",
    r"\bfalse[- ]positive\b": "false_positive",
    r"\bfalse[- ]negative\b": "false_negative",
    r"\benergy[- ]aware\b": "energy_aware",
    r"\bresource[- ]constrained\b": "resource_constrained",
    r"\bconcept drift\b": "concept_drift",
}


def document_id(document: dict[str, Any]) -> str:
    return str(
        document.get("paperId")
        or document.get("paper_id")
        or document.get("id")
        or ""
    )


def limitation_tokens(value: Any) -> set[str]:
    """Return deterministic semantic anchors for one limitation phrase."""
    text = str(value or "").casefold()
    for pattern, replacement in _PHRASES.items():
        text = re.sub(pattern, replacement, text)
    raw = re.findall(r"[a-z0-9_]+", text)
    tokens = {
        _ALIASES.get(token, token)
        for token in raw
        if token not in _GENERIC and len(token) > 2
    }
    return tokens


def split_limitation_dimensions(value: Any) -> list[str]:
    """Split enumerated limitations into atomic, independently clusterable cells."""
    text = re.sub(r"\s+", " ", str(value or "")).strip(" ,;:-")
    text = re.sub(
        r"^(?:of|a lack of|lack of|need for|requirement for)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    comma_parts = re.split(r"\s*;\s*|\s*,\s*(?:and\s+)?", text)
    parts = []
    for part in comma_parts:
        if re.match(
            r"^(?:which|that|thereby|thus|limiting|making|leading|resulting)\b",
            part.strip(),
            flags=re.I,
        ):
            parts.append(part)
            continue
        conjunction_parts = re.split(r"\s+and\s+", part, flags=re.I)
        token_parts = [limitation_tokens(value) for value in conjunction_parts]
        atomic_conjunction = (
            len(conjunction_parts) > 1
            and all(len(tokens) >= 2 for tokens in token_parts)
        )
        zero_day_conjunction = (
            len(conjunction_parts) == 2
            and any("zero_day" in tokens for tokens in token_parts)
        )
        if atomic_conjunction or zero_day_conjunction:
            parts.extend(conjunction_parts)
        else:
            parts.append(part)
    cleaned = []
    for part in parts:
        part = re.sub(r"^(?:and|or|as well as)\s+", "", part, flags=re.I)
        part = part.strip(" ,;:-")
        lowered = part.casefold()
        dependent_fragment = bool(re.match(
            r"^(?:which|that|thereby|thus|limiting|making|leading|resulting)\b",
            lowered,
        ))
        impact_statement = bool(re.search(
            r"\b(?:makes? users?|limits? (?:their )?adoption|challenging to develop)\b",
            lowered,
        ))
        incomplete_adjective = bool(re.search(
            r"\b(?:standardized|standardised|secure|offe)$", lowered
        ))
        tokens = limitation_tokens(part)
        single_token_allowed = bool(tokens.intersection({
            "explainability", "privacy", "scalability", "robustness",
            "generalization", "efficiency",
        }))
        if (
            not dependent_fragment
            and not impact_statement
            and not incomplete_adjective
            and (len(tokens) >= 2 or single_token_allowed)
        ):
            cleaned.append(part)
    return cleaned


def _cell_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left.intersection(right))
    if not overlap:
        return 0.0
    containment = overlap / min(len(left), len(right))
    jaccard = overlap / len(left.union(right))
    return 0.7 * containment + 0.3 * jaccard


def _load_documents(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(config.get("paths", {}).get("processed_data", "")) / "corpus_filtered.jsonl"
    return list(load_jsonl(path)) if path.exists() else []


def _has_full_text(document: dict[str, Any], minimum: int) -> bool:
    return any(
        len(str(document.get(field) or "").strip()) >= minimum
        for field in ("full_text", "fulltext", "text", "content")
    )


def _research_frame(
    G: nx.Graph,
    paper_ids: set[str],
    limitation_nodes: set[str],
) -> dict[str, list[str]]:
    buckets: dict[str, Counter[str]] = {
        "methods": Counter(), "datasets": Counter(), "metrics": Counter(),
        "concepts": Counter(),
    }
    type_to_bucket = {
        "METHOD": "methods", "DATASET": "datasets", "METRIC": "metrics",
        "CONCEPT": "concepts",
    }
    edges = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    for edge in edges:
        left, right, data = str(edge[0]), str(edge[1]), edge[-1]
        paper_id = str(
            data.get("source_paper") or data.get("source_paper_id") or ""
        )
        if paper_id not in paper_ids:
            continue
        for node in (left, right):
            if node in limitation_nodes or not G.has_node(node):
                continue
            bucket = type_to_bucket.get(str(G.nodes[node].get("type", "")).upper())
            if bucket:
                buckets[bucket][node] += 1
    return {
        name: [label for label, _ in counts.most_common(8)]
        for name, counts in buckets.items()
    }


def consolidate_evidence_candidates(
    G: nx.Graph,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], nx.MultiDiGraph]:
    """Merge synonymous limitations and construct a paper-centred evidence map."""
    settings = config.get("gap_detection", {}).get("evidence", {})
    threshold = float(settings.get("cell_similarity_threshold", 0.72))
    minimum_full_text = int(
        config.get("gap_certification", {}).get("min_full_text_characters", 1000)
    )
    domain = str(config.get("project", {}).get("domain", "")).strip()
    documents = _load_documents(config)
    paper_index = {document_id(paper): paper for paper in documents if document_id(paper)}

    claims: list[dict[str, Any]] = []
    for candidate in candidates:
        subject = str(candidate.get("subject", "")).strip()
        for capability in split_limitation_dimensions(
            candidate.get("missing_capability", "")
        ):
            tokens = limitation_tokens(capability)
            if not tokens:
                continue
            for evidence in candidate.get("source_evidence", []):
                claims.append({
                    **evidence,
                    "source_subject": subject,
                    "atomic_limitation": capability,
                    "tokens": tokens,
                    "semantic_scope": evidence.get(
                        "semantic_scope", candidate.get("semantic_scope", "subject")
                    ),
                })

    parent = list(range(len(claims)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(claims)):
        for right in range(left + 1, len(claims)):
            if _cell_similarity(claims[left]["tokens"], claims[right]["tokens"]) >= threshold:
                union(left, right)

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, claim in enumerate(claims):
        groups[find(index)].append(claim)

    evidence_graph = nx.MultiDiGraph(graph_kind="research_gap_evidence_map")
    consolidated: list[dict[str, Any]] = []
    for records in groups.values():
        unique_records = {}
        for record in records:
            key = (
                str(record.get("paper_id", "")),
                str(record.get("source_subject", "")),
                str(record.get("atomic_limitation", "")),
            )
            unique_records.setdefault(key, record)
        records = list(unique_records.values())
        paper_ids = sorted({str(record.get("paper_id")) for record in records if record.get("paper_id")})
        if not paper_ids:
            continue
        token_counts = Counter(token for record in records for token in record["tokens"])
        anchor_tokens = {
            token for token, count in token_counts.items()
            if count >= max(1, (len(records) + 1) // 2)
        } or set(token_counts)
        representative = min(
            (str(record["atomic_limitation"]) for record in records),
            key=lambda value: (len(limitation_tokens(value) - anchor_tokens), len(value)),
        )
        subjects = Counter(str(record.get("source_subject", "")) for record in records)
        subject = subjects.most_common(1)[0][0]
        heterogeneous_subjects = len(subjects) > 1
        semantic_scope = "field" if heterogeneous_subjects else (
            "field" if any(record.get("semantic_scope") == "field" for record in records)
            else "subject"
        )
        if heterogeneous_subjects and domain:
            subject = domain
        limitation_nodes = {
            str(record.get("atomic_limitation", "")) for record in records
        }
        frame = _research_frame(G, set(paper_ids), limitation_nodes)
        frame_completeness = sum(bool(frame[key]) for key in ("methods", "datasets", "metrics")) / 3
        full_text_count = sum(
            _has_full_text(paper_index.get(paper_id, {}), minimum_full_text)
            for paper_id in paper_ids
        )
        mean_confidence = sum(float(record.get("confidence", 0.0)) for record in records) / len(records)
        cohesion_pairs = [
            _cell_similarity(records[left]["tokens"], records[right]["tokens"])
            for left in range(len(records))
            for right in range(left + 1, len(records))
        ]
        cohesion = sum(cohesion_pairs) / len(cohesion_pairs) if cohesion_pairs else 1.0
        source_score = min(len(paper_ids) / 3, 1.0)
        full_text_coverage = full_text_count / len(paper_ids)
        context_score = min(
            sum(bool(frame[key]) for key in ("methods", "datasets", "metrics", "concepts")) / 3,
            1.0,
        )
        quality_score = (
            0.30 * source_score
            + 0.20 * mean_confidence
            + 0.20 * cohesion
            + 0.15 * context_score
            + 0.15 * full_text_coverage
        )
        identity = "|".join(sorted(anchor_tokens)) + "|" + domain.casefold()
        cell_id = "cell_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        source_evidence = [
            {key: value for key, value in record.items() if key != "tokens"}
            for record in records
        ]
        years = [
            int(record["year"]) for record in records
            if str(record.get("year", "")).isdigit()
        ]
        candidate = {
            "type": "evidence_gap",
            "evidence_cell_id": cell_id,
            "subject": subject,
            "missing_capability": representative,
            "canonical_limitation_tokens": sorted(anchor_tokens),
            "limitation_variants": sorted({record["atomic_limitation"] for record in records}),
            "key_concepts": [subject, representative],
            "supporting_paper_ids": paper_ids,
            "supporting_paper_count": len(paper_ids),
            "source_evidence": source_evidence,
            "semantic_scope": semantic_scope,
            "domain": domain,
            "mean_evidence_confidence": round(mean_confidence, 4),
            "first_reported_year": min(years) if years else None,
            "latest_reported_year": max(years) if years else None,
            "research_frame": frame,
            "candidate_quality": {
                "score": round(quality_score, 4),
                "independent_source_count": len(paper_ids),
                "semantic_cohesion": round(cohesion, 4),
                "frame_completeness": round(frame_completeness, 4),
                "full_text_source_count": full_text_count,
                "full_text_coverage": round(full_text_coverage, 4),
                "certificate_readiness": {
                    "independent_sources_ready": len(paper_ids) >= int(
                        config.get("gap_certification", {}).get(
                            "min_independent_limitation_sources", 2
                        )
                    ),
                    "full_text_ready": full_text_count == len(paper_ids),
                    "research_frame_ready": frame_completeness >= 2 / 3,
                },
            },
            "description": (
                f"Evidence cell '{representative}' is reported by {len(paper_ids)} "
                f"independent paper(s) in {domain or 'the screened domain'}."
            ),
            "draft_claim": (
                f"Within the screened corpus, independent evidence reports "
                f"'{representative}' as an unresolved limitation."
            ),
        }
        consolidated.append(candidate)

        evidence_graph.add_node(
            cell_id,
            type="EVIDENCE_CELL",
            label=representative,
            source_count=len(paper_ids),
            quality_score=round(quality_score, 4),
        )
        for paper_id in paper_ids:
            paper = paper_index.get(paper_id, {})
            paper_node = f"paper::{paper_id}"
            evidence_graph.add_node(
                paper_node,
                type="PAPER",
                label=str(paper.get("title") or paper_id),
                year=paper.get("year") or "",
            )
            evidence_graph.add_edge(
                paper_node, cell_id, relation="REPORTS_LIMITATION"
            )
        for frame_name, node_type in (
            ("methods", "METHOD"), ("datasets", "DATASET"),
            ("metrics", "METRIC"), ("concepts", "CONCEPT"),
        ):
            for label in frame[frame_name][:5]:
                context_node = f"context::{node_type}::{label}"
                evidence_graph.add_node(context_node, type=node_type, label=label)
                evidence_graph.add_edge(
                    cell_id, context_node, relation=f"HAS_{node_type}_CONTEXT"
                )

    consolidated.sort(
        key=lambda item: (
            item["candidate_quality"]["score"],
            item["supporting_paper_count"],
            item["mean_evidence_confidence"],
        ),
        reverse=True,
    )
    return consolidated, evidence_graph


def save_evidence_graph(
    graph: nx.MultiDiGraph,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    graph_dir_value = config.get("paths", {}).get("graph")
    if not graph_dir_value:
        return
    graph_dir = ensure_dir(graph_dir_value)
    with open(Path(graph_dir) / "gap_evidence_graph.pkl", "wb") as stream:
        pickle.dump(graph, stream)
    graphml = graph.copy()
    nx.write_graphml(graphml, Path(graph_dir) / "gap_evidence_graph.graphml")
    save_json(candidates, Path(graph_dir) / "evidence_cells.json")


def add_empty_cells_to_evidence_graph(
    candidates: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    """Attach typed empty evidence-map cells to the paper-centred graph."""
    graph_dir_value = config.get("paths", {}).get("graph")
    if not graph_dir_value:
        return
    graph_dir = Path(graph_dir_value)
    pickle_path = graph_dir / "gap_evidence_graph.pkl"
    if pickle_path.exists():
        with open(pickle_path, "rb") as stream:
            graph = pickle.load(stream)
    else:
        graph = nx.MultiDiGraph(graph_kind="research_gap_evidence_map")
    paper_index = {
        document_id(paper): paper for paper in _load_documents(config)
        if document_id(paper)
    }
    for candidate in candidates:
        head, tail = str(candidate.get("head", "")), str(candidate.get("tail", ""))
        if not head or not tail:
            continue
        identity = "|".join(sorted((head.casefold(), tail.casefold())))
        cell_id = "empty_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        quality = float(candidate.get("candidate_quality", {}).get("score", 0.0))
        graph.add_node(
            cell_id,
            type="EVIDENCE_CELL",
            label=f"{head} × {tail}",
            cell_kind="TYPED_EMPTY_CELL",
            source_count=len(candidate.get("supporting_paper_ids", [])),
            quality_score=quality,
        )
        frame = candidate.get("evidence_map", {}) or {}
        for label, node_type in (
            (head, frame.get("head_type", "CONCEPT")),
            (tail, frame.get("tail_type", "CONCEPT")),
        ):
            context_node = f"context::{node_type}::{label}"
            graph.add_node(context_node, type=node_type, label=label)
            graph.add_edge(cell_id, context_node, relation="DEFINES_EMPTY_CELL")
        for paper_id in candidate.get("supporting_paper_ids", []):
            paper_id = str(paper_id)
            paper = paper_index.get(paper_id, {})
            paper_node = f"paper::{paper_id}"
            graph.add_node(
                paper_node,
                type="PAPER",
                label=str(paper.get("title") or paper_id),
                year=paper.get("year") or "",
            )
            graph.add_edge(paper_node, cell_id, relation="SUPPORTS_EMPTY_CELL_PATH")
    with open(pickle_path, "wb") as stream:
        pickle.dump(graph, stream)
    nx.write_graphml(graph, graph_dir / "gap_evidence_graph.graphml")
