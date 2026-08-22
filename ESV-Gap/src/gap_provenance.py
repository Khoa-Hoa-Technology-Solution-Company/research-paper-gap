"""Resolve research-gap candidates back to their source evidence papers.

Gap candidates are graph-level inferences.  The papers returned here support
the graph relations that motivated a candidate; they do not necessarily state
the candidate gap explicitly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import networkx as nx


PAPER_ID_KEYS = ("source_paper", "source_paper_id", "paper_id")


def build_paper_index(papers: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index corpus metadata by the paper identifier stored on graph edges."""
    index: dict[str, dict[str, Any]] = {}
    for paper in papers:
        paper_id = paper.get("paperId") or paper.get("paper_id") or paper.get("id")
        if paper_id:
            index[str(paper_id)] = paper
    return index


def candidate_identity(candidate: dict[str, Any]) -> str:
    """Return a stable identity for caching provenance in the UI."""
    gap_type = str(candidate.get("type", "unknown"))
    if gap_type == "missing_link":
        details = (
            str(candidate.get("head", "")),
            str(candidate.get("relation", "")),
            str(candidate.get("tail", "")),
        )
    elif gap_type == "orphan_cluster":
        details = tuple(sorted(map(str, candidate.get("members", []))))
    elif gap_type == "temporal_decay":
        details = (str(candidate.get("concept", "")),)
    else:
        details = (str(candidate.get("description", "")),)
    return "|".join((gap_type, *details))


def _paper_id(data: dict[str, Any]) -> str | None:
    for key in PAPER_ID_KEYS:
        if data.get(key):
            return str(data[key])
    return None


def _edge_records_between(G: nx.Graph, left: str, right: str) -> list[dict[str, Any]]:
    """Return directed relation events connecting two adjacent concepts."""
    pairs = [(left, right)]
    if G.is_directed():
        pairs.append((right, left))

    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source, target in pairs:
        if not G.has_edge(source, target):
            continue
        if G.is_multigraph():
            edge_data = G.get_edge_data(source, target, default={}).values()
        else:
            edge_data = [G.get_edge_data(source, target, default={})]
        for data in edge_data:
            record = {
                "subject": str(source),
                "relation": str(data.get("relation", "RELATED_TO")),
                "object": str(target),
                "paper_id": _paper_id(data),
                "year": data.get("year"),
                "evidence": str(data.get("evidence", "") or ""),
            }
            signature = tuple(record.get(key) for key in (
                "subject", "relation", "object", "paper_id", "evidence"
            ))
            if signature not in seen:
                records.append(record)
                seen.add(signature)
    return records


def _missing_link_records(
    G: nx.Graph,
    candidate: dict[str, Any],
    max_paths: int,
    cutoff: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    head = str(candidate.get("head", ""))
    tail = str(candidate.get("tail", ""))
    simple = nx.Graph(G)
    if not head or not tail or head == tail or head not in simple or tail not in simple:
        return [], []

    paths: list[list[str]] = []
    validation = candidate.get("validation", {}) or {}
    supplied_paths = (
        candidate.get("independent_evidence_paths", [])
        or validation.get("independent_evidence_paths", [])
    )
    for item in supplied_paths:
        nodes = item.get("nodes", []) if isinstance(item, dict) else []
        if len(nodes) >= 2:
            paths.append(list(map(str, nodes)))

    if not paths:
        try:
            for path in nx.shortest_simple_paths(simple, head, tail):
                if len(path) - 1 > cutoff:
                    break
                paths.append(list(map(str, path)))
                if len(paths) >= max_paths:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], []

    records: list[dict[str, Any]] = []
    path_details: list[dict[str, Any]] = []
    for path in paths[:max_paths]:
        path_records: list[dict[str, Any]] = []
        for left, right in zip(path, path[1:]):
            path_records.extend(_edge_records_between(G, left, right))
        paper_ids = sorted({
            record["paper_id"] for record in path_records if record.get("paper_id")
        })
        path_details.append({"nodes": path, "paper_ids": paper_ids})
        records.extend(path_records)
    return records, path_details


def _orphan_records(G: nx.Graph, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    members = set(map(str, candidate.get("members", [])))
    if not members:
        return []
    records: list[dict[str, Any]] = []
    graph_edges = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    for edge in graph_edges:
        source, target, data = edge[0], edge[1], edge[-1]
        if str(source) in members and str(target) in members:
            records.append({
                "subject": str(source),
                "relation": str(data.get("relation", "RELATED_TO")),
                "object": str(target),
                "paper_id": _paper_id(data),
                "year": data.get("year"),
                "evidence": str(data.get("evidence", "") or ""),
            })
    return records


def _temporal_records(G: nx.Graph, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    concept = str(candidate.get("concept", ""))
    if not concept or not G.has_node(concept):
        return []
    records: list[dict[str, Any]] = []
    neighbours = set(G.successors(concept)) | set(G.predecessors(concept)) \
        if G.is_directed() else set(G.neighbors(concept))
    for neighbour in sorted(neighbours, key=str):
        records.extend(_edge_records_between(G, concept, str(neighbour)))
    return records


def _normalise_paper(paper_id: str, paper: dict[str, Any]) -> dict[str, Any]:
    authors = []
    for author in paper.get("authors", []) or []:
        name = author.get("name") if isinstance(author, dict) else str(author)
        if name:
            authors.append(str(name))
    external_ids = paper.get("externalIds", {}) or {}
    screening = paper.get("screening", {}) or {}
    return {
        "paper_id": paper_id,
        "title": str(paper.get("title") or f"Paper {paper_id}"),
        "authors": authors,
        "year": paper.get("year"),
        "venue": str(paper.get("venue") or ""),
        "citation_count": paper.get("citationCount", 0),
        "url": str(paper.get("url") or ""),
        "doi": str(external_ids.get("DOI") or ""),
        "abstract": str(paper.get("abstract") or ""),
        "relevance_score": screening.get("confidence"),
        "relevance_reason": str(screening.get("reason") or ""),
    }


def resolve_gap_provenance(
    G: nx.Graph | None,
    candidate: dict[str, Any],
    paper_index: dict[str, dict[str, Any]],
    max_paths: int = 3,
    cutoff: int = 4,
) -> dict[str, Any]:
    """Resolve a candidate to papers and edge-level evidence records."""
    if G is None:
        return {"paper_ids": [], "papers": [], "evidence_paths": []}

    gap_type = candidate.get("type")
    path_details: list[dict[str, Any]] = []
    if gap_type == "missing_link":
        records, path_details = _missing_link_records(G, candidate, max_paths, cutoff)
    elif gap_type == "orphan_cluster":
        records = _orphan_records(G, candidate)
    elif gap_type == "temporal_decay":
        records = _temporal_records(G, candidate)
    else:
        records = []

    validation = candidate.get("validation", {}) or {}
    explicit_ids = (
        candidate.get("supporting_paper_ids", [])
        or candidate.get("_supporting_paper_ids", [])
        or validation.get("supporting_paper_ids", [])
    )
    paper_ids = {str(value) for value in explicit_ids if value}
    paper_ids.update(record["paper_id"] for record in records if record.get("paper_id"))

    evidence_by_paper: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_by_paper: defaultdict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for record in records:
        paper_id = record.get("paper_id")
        if not paper_id:
            continue
        signature = tuple(record.get(key) for key in (
            "subject", "relation", "object", "evidence"
        ))
        if signature not in seen_by_paper[paper_id]:
            evidence_by_paper[paper_id].append(record)
            seen_by_paper[paper_id].add(signature)

    papers = []
    for paper_id in sorted(paper_ids):
        paper = _normalise_paper(paper_id, paper_index.get(paper_id, {}))
        paper["evidence"] = evidence_by_paper.get(paper_id, [])
        papers.append(paper)
    papers.sort(key=lambda item: (
        -(int(item.get("year") or 0)),
        item.get("title", "").lower(),
    ))

    return {
        "paper_ids": [paper["paper_id"] for paper in papers],
        "papers": papers,
        "evidence_paths": path_details,
    }
