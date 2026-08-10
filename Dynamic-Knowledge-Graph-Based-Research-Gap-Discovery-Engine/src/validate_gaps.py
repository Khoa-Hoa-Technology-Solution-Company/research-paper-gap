"""Stage 5: evidence- and perturbation-aware triage of gap candidates.

The detectors in :mod:`src.detect_gaps` produce *signals*, not validated
research gaps.  This module inserts a conservative validation gate before
ranking.  It rejects candidates that are generic, lack path-specific
multi-paper provenance, or are unstable under several plausible knowledge-
graph perturbations.  Existing relations and lexical coverage hits are routed
to review: neither is treated as proof that a scientific gap is closed.

The implementation is deliberately deterministic.  Candidate-specific
bootstrap seeds are derived from SHA-256 hashes, so repeated runs over the
same graph and configuration produce identical decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from src.entity_normalization import canonical_entity_key, canonical_entity_label
from src.utils import ensure_dir, get_logger, load_json, save_json


logger = get_logger("validate_gaps")

GENERIC_PHRASES = {
    "algorithm",
    "analysis",
    "approach",
    "concept",
    "framework",
    "method",
    "model",
    "our approach",
    "our framework",
    "our method",
    "proposed approach",
    "proposed framework",
    "proposed method",
    "research",
    "system",
    "technique",
    "this paper",
    "this work",
}

STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "using", "via", "with",
}


def _tokens(text: str) -> set[str]:
    output: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", canonical_entity_label(text).lower()):
        if token in STOPWORDS:
            continue
        output.add(token)
        versionless = re.sub(r"\d+$", "", token)
        if versionless and versionless != token and versionless not in STOPWORDS:
            output.add(versionless)
    return output


def _stable_seed(candidate: dict[str, Any], base_seed: int) -> int:
    identity = json.dumps(
        {
            "type": candidate.get("type"),
            "head": candidate.get("head"),
            "tail": candidate.get("tail"),
            "concept": candidate.get("concept"),
            "community_id": candidate.get("community_id"),
            "members": sorted(map(str, candidate.get("members", []))),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def candidate_entities(candidate: dict[str, Any], limit: int = 12) -> list[str]:
    """Return the entities that define a detector candidate."""
    gap_type = candidate.get("type")
    if gap_type == "missing_link":
        values = [candidate.get("head"), candidate.get("tail")]
    elif gap_type == "temporal_decay":
        values = [candidate.get("concept")]
    else:
        values = candidate.get("key_concepts") or candidate.get("members") or []

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = canonical_entity_label(value)
        key = canonical_entity_key(value)
        if value and key not in seen:
            output.append(value)
            seen.add(key)
        if len(output) >= limit:
            break
    return output


def specificity_score(entities: Iterable[str]) -> float:
    """Estimate whether entity labels are domain-specific rather than placeholders."""
    values = [
        canonical_entity_label(value).lower()
        for value in entities
        if canonical_entity_label(value)
    ]
    if not values:
        return 0.0

    scores = []
    for value in values:
        tokens = _tokens(value)
        if value in GENERIC_PHRASES or not tokens:
            scores.append(0.0)
            continue
        generic_tokens = sum(token in GENERIC_PHRASES for token in tokens)
        lexical = 1.0 - generic_tokens / max(len(tokens), 1)
        length_bonus = min(len(tokens) / 3.0, 1.0)
        scores.append(0.75 * lexical + 0.25 * length_bonus)
    return round(sum(scores) / len(scores), 4)


def _iter_edge_data(G: nx.Graph, node: str):
    if not G.has_node(node):
        return
    if G.is_multigraph():
        for _, _, _, data in G.edges(node, keys=True, data=True):
            yield data
        if G.is_directed():
            for _, _, _, data in G.in_edges(node, keys=True, data=True):
                yield data
    else:
        for _, _, data in G.edges(node, data=True):
            yield data
        if G.is_directed():
            for _, _, data in G.in_edges(node, data=True):
                yield data


def evidence_papers(G: nx.Graph, entities: Iterable[str]) -> set[str]:
    """Collect unique source-paper identifiers attached to nodes and edges."""
    papers: set[str] = set()
    for entity in entities:
        if not G.has_node(entity):
            continue
        node_papers = G.nodes[entity].get("papers", [])
        if isinstance(node_papers, str):
            node_papers = [part.strip() for part in node_papers.split(",")]
        papers.update(str(value) for value in node_papers if value)
        for data in _iter_edge_data(G, entity):
            for key in ("source_paper", "source_paper_id", "paper_id"):
                if data.get(key):
                    papers.add(str(data[key]))
    return papers


def community_internal_papers(G: nx.Graph, members: Iterable[str]) -> set[str]:
    """Return only papers supporting relations inside a candidate community."""
    member_set = set(map(str, members))
    papers: set[str] = set()
    records = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    for record in records:
        u, v, data = record[0], record[1], record[-1]
        if str(u) not in member_set or str(v) not in member_set:
            continue
        paper = data.get("source_paper") or data.get("source_paper_id") or data.get("paper_id")
        if paper:
            papers.add(str(paper))
    return papers


def _edge_papers(G: nx.Graph, u: str, v: str) -> set[str]:
    """Return provenance IDs on all relation events between two nodes."""
    papers: set[str] = set()
    if not G.has_edge(u, v) and not G.has_edge(v, u):
        return papers
    pairs = ((u, v), (v, u)) if G.is_directed() else ((u, v),)
    for source, target in pairs:
        if not G.has_edge(source, target):
            continue
        if G.is_multigraph():
            records = G.get_edge_data(source, target, default={}).values()
        else:
            records = [G.get_edge_data(source, target, default={})]
        for data in records:
            for key in ("source_paper", "source_paper_id", "paper_id"):
                if data.get(key):
                    papers.add(str(data[key]))
    return papers


def independent_evidence_paths(
    G: nx.Graph,
    head: str,
    tail: str,
    cutoff: int = 4,
) -> list[dict[str, Any]]:
    """Find short paths that are both edge-disjoint and source-disjoint.

    Candidate provenance is attached to the paths that motivate the missing
    relation, rather than to every incident edge of either endpoint.  The
    greedy selection is deterministic (shortest path, then lexical order).
    """
    simple = _simple_undirected(G)
    if head not in simple or tail not in simple or head == tail:
        return []
    paths = list(nx.all_simple_paths(simple, head, tail, cutoff=cutoff))
    paths.sort(key=lambda path: (len(path), tuple(map(str, path))))
    selected: list[dict[str, Any]] = []
    used_edges: set[tuple[str, str]] = set()
    used_papers: set[str] = set()
    for path in paths:
        edges = {
            tuple(sorted((str(u), str(v))))
            for u, v in zip(path, path[1:])
        }
        papers: set[str] = set()
        fully_traceable = True
        for u, v in zip(path, path[1:]):
            edge_sources = _edge_papers(G, u, v)
            if not edge_sources:
                fully_traceable = False
            papers.update(edge_sources)
        if not fully_traceable or edges.intersection(used_edges) or papers.intersection(used_papers):
            continue
        selected.append({"nodes": list(map(str, path)), "papers": sorted(papers)})
        used_edges.update(edges)
        used_papers.update(papers)
    return selected


def _simple_undirected(G: nx.Graph) -> nx.Graph:
    simple = nx.Graph()
    simple.add_nodes_from(G.nodes(data=True))
    simple.add_edges_from((u, v) for u, v in G.edges() if u != v)
    return simple


def _edge_dropout_graph(graph: nx.Graph, rng: random.Random, keep: float) -> nx.Graph:
    sampled = graph.copy()
    if sampled.is_multigraph():
        removals = [
            (u, v, key)
            for u, v, key in sampled.edges(keys=True)
            if rng.random() > keep
        ]
    else:
        removals = [(u, v) for u, v in sampled.edges() if rng.random() > keep]
    sampled.remove_edges_from(removals)
    return sampled


def _paper_dropout_graph(graph: nx.Graph, rng: random.Random, keep: float) -> nx.Graph:
    """Drop papers as clusters, removing every relation event they generated."""
    papers: set[str] = set()
    edge_records = graph.edges(keys=True, data=True) if graph.is_multigraph() else graph.edges(data=True)
    for record in edge_records:
        data = record[-1]
        paper = data.get("source_paper") or data.get("source_paper_id") or data.get("paper_id")
        if paper:
            papers.add(str(paper))
    retained = {paper for paper in sorted(papers) if rng.random() <= keep}
    sampled = graph.copy()
    edge_records = list(sampled.edges(keys=True, data=True)) if sampled.is_multigraph() else list(sampled.edges(data=True))
    removals = []
    for record in edge_records:
        data = record[-1]
        paper = data.get("source_paper") or data.get("source_paper_id") or data.get("paper_id")
        if not paper or str(paper) not in retained:
            removals.append(record[:-1])
    sampled.remove_edges_from(removals)
    return sampled


def _add_plausible_edges(
    graph: nx.Graph,
    candidate: dict[str, Any],
    rng: random.Random,
    probability: float,
) -> nx.Graph:
    """Sample candidate edges supplied by retrieval/link-prediction evidence."""
    sampled = graph.copy()
    for index, item in enumerate(candidate.get("plausible_edges", [])):
        if rng.random() > probability:
            continue
        if isinstance(item, dict):
            u, v = item.get("head"), item.get("tail")
            attrs = {
                "relation": item.get("relation", "PLAUSIBLE"),
                "source_paper": item.get("paper_id", f"plausible-{index}"),
                "year": item.get("year"),
                "confidence": item.get("confidence", 0.0),
            }
        else:
            try:
                u, v = item[:2]
            except (TypeError, ValueError):
                continue
            attrs = {"relation": "PLAUSIBLE", "source_paper": f"plausible-{index}"}
        if u and v and u != v:
            sampled.add_edge(str(u), str(v), **attrs)
    return sampled


def _stress_relevant_plausible_edges(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only plausible edges that can challenge this candidate.

    An empty result is not by itself evidence that the search ran.  The
    validator requires a separate completion marker before treating an empty
    result as "searched, no relevant closing edge found."
    """
    normalised: list[dict[str, Any]] = []
    for index, item in enumerate(candidate.get("plausible_edges", [])):
        if isinstance(item, dict):
            u, v = str(item.get("head", "")).strip(), str(item.get("tail", "")).strip()
            edge = dict(item)
        else:
            try:
                u, v = (str(value).strip() for value in item[:2])
            except (TypeError, ValueError):
                continue
            edge = {"head": u, "tail": v, "paper_id": f"plausible-{index}"}
        if not u or not v or u == v:
            continue
        edge["head"], edge["tail"] = u, v
        normalised.append(edge)

    gap_type = candidate.get("type")
    if gap_type == "missing_link":
        endpoints = {str(candidate.get("head", "")), str(candidate.get("tail", ""))}
        return [edge for edge in normalised if {edge["head"], edge["tail"]} == endpoints]
    if gap_type == "orphan_cluster":
        members = set(map(str, candidate.get("members", [])))
        return [
            edge for edge in normalised
            if (edge["head"] in members) != (edge["tail"] in members)
        ]
    if gap_type == "temporal_decay":
        concept = str(candidate.get("concept", ""))
        return [edge for edge in normalised if concept in {edge["head"], edge["tail"]}]
    return normalised


def _plausible_stress_available(
    candidate: dict[str, Any],
    relevant_edges: list[dict[str, Any]],
) -> bool:
    """Distinguish an executed search with zero hits from a missing search.

    A non-empty relevant pool proves that a search result is present.  An
    empty pool is evaluable only when its producer explicitly records that the
    candidate-level plausible-edge search completed.
    """
    return bool(
        relevant_edges
        or candidate.get("plausible_edge_search_performed") is True
        or candidate.get("plausible_edges_evaluated") is True
    )


def _orphan_isolation(graph: nx.Graph, members: set[str]) -> float:
    members = members.intersection(graph.nodes())
    if len(members) < 2:
        return 0.0
    internal = graph.subgraph(members).number_of_edges()
    cut = nx.cut_size(graph, members)
    return 1.0 - cut / max(cut + internal, 1)


def _temporal_decay(
    G: nx.Graph,
    concept: str,
    lookback: int,
    publication_counts: dict[int, int] | None = None,
    analysis_end_year: int | None = None,
) -> tuple[float, int]:
    papers_by_year: defaultdict[int, set[str]] = defaultdict(set)
    for data in _iter_edge_data(G, concept):
        year = data.get("year")
        try:
            if year is not None and str(year).strip():
                paper = data.get("source_paper") or data.get("source_paper_id") or data.get("paper_id")
                papers_by_year[int(year)].add(str(paper or id(data)))
        except (TypeError, ValueError):
            continue
    if not papers_by_year:
        return 0.0, 0
    max_year = min(analysis_end_year or max(papers_by_year), max(papers_by_year))
    totals = publication_counts or {
        year: max(len(papers), 1) for year, papers in papers_by_year.items()
    }
    rates = {
        year: len(papers_by_year.get(year, set())) / max(int(totals.get(year, 0)), 1)
        for year in range(max_year - 2 * lookback + 1, max_year + 1)
    }
    recent = sum(rates[y] for y in range(max_year - lookback + 1, max_year + 1)) / max(lookback, 1)
    earlier = sum(rates[y] for y in range(max_year - 2 * lookback + 1, max_year - lookback + 1)) / max(lookback, 1)
    decay = 1.0 - recent / earlier if earlier else 0.0
    return max(0.0, min(decay, 1.0)), len(papers_by_year)


def _candidate_survives(
    graph: nx.Graph,
    candidate: dict[str, Any],
    settings: dict[str, Any],
) -> bool:
    gap_type = candidate.get("type")
    if gap_type == "missing_link":
        head, tail = str(candidate.get("head", "")), str(candidate.get("tail", ""))
        if not head or not tail or graph.has_edge(head, tail) or graph.has_edge(tail, head):
            return False
        paths = independent_evidence_paths(
            graph, head, tail, cutoff=int(settings.get("max_path_length", 4))
        )
        return len(paths) >= int(settings.get("min_surviving_paths", 1))
    if gap_type == "orphan_cluster":
        members = set(map(str, candidate.get("members", [])))
        return _orphan_isolation(_simple_undirected(graph), members) >= float(
            settings["orphan_isolation_threshold"]
        )
    if gap_type == "temporal_decay":
        counts = {
            int(year): int(count)
            for year, count in candidate.get("publication_counts", {}).items()
        }
        decay, distinct_years = _temporal_decay(
            graph,
            str(candidate.get("concept", "")),
            int(settings["temporal_lookback_years"]),
            publication_counts=counts or None,
            analysis_end_year=candidate.get("analysis_end_year"),
        )
        return decay >= float(settings["temporal_decay_threshold"]) and distinct_years >= 2
    return False


def _orphan_perturbation_scores(
    G: nx.Graph,
    candidate: dict[str, Any],
    settings: dict[str, Any],
    repeats: int,
    base_seed: int,
) -> dict[str, float | None]:
    """Fast sufficient-statistic bootstrap for an orphan community."""
    simple = _simple_undirected(G)
    members = set(map(str, candidate.get("members", []))).intersection(simple.nodes())
    internal = [tuple(sorted((str(u), str(v)))) for u, v in simple.subgraph(members).edges()]
    cut = []
    for member in members:
        for neighbour in simple.neighbors(member):
            if neighbour not in members:
                cut.append(tuple(sorted((str(member), str(neighbour)))))
    cut = sorted(set(cut))
    threshold = float(settings["orphan_isolation_threshold"])

    def isolated(internal_count: int, cut_count: int) -> bool:
        return 1.0 - cut_count / max(internal_count + cut_count, 1) >= threshold

    edge_rng = random.Random(base_seed)
    edge_success = 0
    edge_keep = float(settings.get("edge_keep_probability", 0.95))
    for _ in range(repeats):
        kept_internal = sum(edge_rng.random() <= edge_keep for _ in internal)
        kept_cut = sum(edge_rng.random() <= edge_keep for _ in cut)
        edge_success += int(isolated(kept_internal, kept_cut))

    relevant_edges = internal + cut
    edge_sources = {edge: _edge_papers(G, *edge) for edge in relevant_edges}
    all_papers = sorted({paper for papers in edge_sources.values() for paper in papers})
    paper_rng = random.Random(base_seed + 1_000_003)
    paper_keep = float(settings.get("paper_keep_probability", 0.95))
    paper_success = 0
    for _ in range(repeats):
        retained = {paper for paper in all_papers if paper_rng.random() <= paper_keep}
        kept_internal = sum(bool(edge_sources[edge].intersection(retained)) for edge in internal)
        kept_cut = sum(bool(edge_sources[edge].intersection(retained)) for edge in cut)
        paper_success += int(isolated(kept_internal, kept_cut))

    addition_rng = random.Random(base_seed + 2_000_006)
    add_probability = float(settings.get("plausible_edge_add_probability", 0.5))
    plausible = _stress_relevant_plausible_edges(candidate)
    addition_available = _plausible_stress_available(candidate, plausible)
    addition_success = 0
    if addition_available:
        for _ in range(repeats):
            added_cut = sum(
                addition_rng.random() <= add_probability for _ in plausible
            )
            addition_success += int(isolated(len(internal), len(cut) + added_cut))

    scores = {
        "edge_deletion": round(edge_success / repeats, 4),
        "paper_dropout": round(paper_success / repeats, 4),
        "plausible_edge_addition": (
            round(addition_success / repeats, 4) if addition_available else None
        ),
    }
    evaluated = [value for value in scores.values() if value is not None]
    scores["aggregate"] = round(min(evaluated), 4)
    return scores


def _missing_link_perturbation_scores(
    G: nx.Graph,
    candidate: dict[str, Any],
    settings: dict[str, Any],
    repeats: int,
    base_seed: int,
) -> dict[str, float | None]:
    """Bootstrap the candidate's traceable paths without copying the full KG."""
    head, tail = str(candidate.get("head", "")), str(candidate.get("tail", ""))
    if G.has_edge(head, tail) or G.has_edge(tail, head):
        relevant = _stress_relevant_plausible_edges(candidate)
        addition = 0.0 if _plausible_stress_available(candidate, relevant) else None
        return {"edge_deletion": 0.0, "paper_dropout": 0.0, "plausible_edge_addition": addition, "aggregate": 0.0}
    paths = candidate.get("independent_evidence_paths") or independent_evidence_paths(
        G, head, tail, cutoff=int(settings.get("max_path_length", 4))
    )
    min_paths = int(settings.get("min_surviving_paths", 1))
    path_edges = [
        {tuple(sorted((u, v))) for u, v in zip(path["nodes"], path["nodes"][1:])}
        for path in paths
    ]
    all_edges = sorted({edge for edges in path_edges for edge in edges})
    edge_rng = random.Random(base_seed)
    edge_keep = float(settings.get("edge_keep_probability", 0.95))
    edge_success = 0
    for _ in range(repeats):
        retained = {edge for edge in all_edges if edge_rng.random() <= edge_keep}
        edge_success += int(sum(edges.issubset(retained) for edges in path_edges) >= min_paths)

    path_papers = [set(map(str, path.get("papers", []))) for path in paths]
    all_papers = sorted({paper for papers in path_papers for paper in papers})
    paper_rng = random.Random(base_seed + 1_000_003)
    paper_keep = float(settings.get("paper_keep_probability", 0.95))
    paper_success = 0
    for _ in range(repeats):
        retained = {paper for paper in all_papers if paper_rng.random() <= paper_keep}
        paper_success += int(sum(papers.issubset(retained) for papers in path_papers) >= min_paths)

    addition_rng = random.Random(base_seed + 2_000_006)
    probability = float(settings.get("plausible_edge_add_probability", 0.5))
    plausible_direct = _stress_relevant_plausible_edges(candidate)
    addition_available = _plausible_stress_available(candidate, plausible_direct)
    addition_success = 0
    if addition_available:
        addition_success = sum(
            int(
                len(paths) >= min_paths
                and not any(addition_rng.random() <= probability for _ in plausible_direct)
            )
            for _ in range(repeats)
        )
    scores = {
        "edge_deletion": round(edge_success / repeats, 4),
        "paper_dropout": round(paper_success / repeats, 4),
        "plausible_edge_addition": (
            round(addition_success / repeats, 4) if addition_available else None
        ),
    }
    evaluated = [value for value in scores.values() if value is not None]
    scores["aggregate"] = round(min(evaluated), 4)
    return scores


def perturbation_stability(
    G: nx.Graph,
    candidate: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, float | None]:
    """Evaluate edge deletion, paper dropout, and plausible edge addition.

    The returned aggregate is the minimum among evaluated modes.  If no
    candidate-relevant plausible edge exists, the addition mode is ``None``;
    validation then routes the candidate to review instead of treating the
    unavailable test as a success.
    """
    repeats = max(int(settings.get("bootstrap_repeats", 100)), 1)
    base_seed = _stable_seed(candidate, int(settings.get("random_seed", 42)))
    if candidate.get("type") == "missing_link":
        return _missing_link_perturbation_scores(G, candidate, settings, repeats, base_seed)
    if candidate.get("type") == "orphan_cluster":
        return _orphan_perturbation_scores(G, candidate, settings, repeats, base_seed)
    modes = {
        "edge_deletion": lambda rng: _edge_dropout_graph(
            G, rng, float(settings.get("edge_keep_probability", 0.8))
        ),
        "paper_dropout": lambda rng: _paper_dropout_graph(
            G, rng, float(settings.get("paper_keep_probability", 0.8))
        ),
    }
    relevant_plausible = _stress_relevant_plausible_edges(candidate)
    addition_available = _plausible_stress_available(candidate, relevant_plausible)
    if addition_available:
        stress_candidate = dict(candidate)
        stress_candidate["plausible_edges"] = relevant_plausible
        modes["plausible_edge_addition"] = lambda rng: _add_plausible_edges(
            G, stress_candidate, rng, float(settings.get("plausible_edge_add_probability", 0.5))
        )
    scores: dict[str, float | None] = {}
    for offset, (name, sampler) in enumerate(modes.items()):
        rng = random.Random(base_seed + 1_000_003 * offset)
        successes = sum(
            int(_candidate_survives(sampler(rng), candidate, settings))
            for _ in range(repeats)
        )
        scores[name] = round(successes / repeats, 4)
    if not addition_available:
        scores["plausible_edge_addition"] = None
    evaluated = [value for value in scores.values() if value is not None]
    scores["aggregate"] = round(min(evaluated), 4)
    return scores


def _document_text(document: dict[str, Any]) -> str:
    return " ".join(
        str(document.get(key, ""))
        for key in ("title", "abstract", "text", "content")
        if document.get(key)
    ).lower()


def closure_hits(
    candidate: dict[str, Any],
    documents: Iterable[dict[str, Any]],
    token_coverage: float = 0.60,
) -> list[dict[str, Any]]:
    """Find local-corpus documents that co-mention the candidate's entity groups.

    This is a conservative *screen*, not proof that a gap is closed.  Hits force
    manual review instead of automatic acceptance.
    """
    entities = candidate_entities(candidate, limit=6)
    aliases = candidate.get("entity_aliases", {})
    groups = []
    for entity in entities:
        alternatives = [_tokens(canonical_entity_label(entity))]
        alias_values = []
        for alias_entity, values in aliases.items():
            if canonical_entity_key(alias_entity) == canonical_entity_key(entity):
                alias_values.extend(values)
        alternatives.extend(_tokens(canonical_entity_label(alias)) for alias in alias_values)
        alternatives = [tokens for tokens in alternatives if tokens]
        if alternatives:
            groups.append(alternatives)
    if not groups:
        return []

    temporal_screen = candidate.get("type") == "temporal_decay" and len(groups) == 1
    if len(groups) < 2 and not temporal_screen:
        return []

    hits = []
    for index, document in enumerate(documents):
        text = _document_text(document)
        if not text:
            continue
        text_tokens = _tokens(text)
        matched = [
            alternatives
            for alternatives in groups
            if any(
                len(tokens.intersection(text_tokens))
                >= max(1, math.ceil(len(tokens) * token_coverage))
                for tokens in alternatives
            )
        ]
        document_year = document.get("year") or document.get("publication_year")
        if temporal_screen:
            try:
                is_hit = len(matched) == 1 and int(document_year) > int(candidate.get("peak_year", 0))
            except (TypeError, ValueError):
                is_hit = False
        else:
            is_hit = len(matched) >= 2
        if is_hit:
            hits.append({
                "paper_id": document.get("paperId") or document.get("paper_id") or document.get("id") or str(index),
                "title": document.get("title", ""),
                "year": document_year,
            })
    return hits


def validate_candidate(
    G: nx.Graph,
    candidate: dict[str, Any],
    config: dict[str, Any],
    documents: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one detector output and return an auditable decision record."""
    settings = config["gap_validation"]
    raw_entities = candidate_entities(candidate)
    entities = candidate_entities(candidate)
    documents_list = None if documents is None else list(documents)
    gap_type = candidate.get("type")
    evidence_paths: list[dict[str, Any]] = []
    if gap_type == "missing_link":
        evidence_paths = independent_evidence_paths(
            G,
            str(candidate.get("head", "")),
            str(candidate.get("tail", "")),
            cutoff=int(settings.get("max_path_length", 4)),
        )
        papers = {
            paper
            for path in evidence_paths
            for paper in path.get("papers", [])
        }
    elif gap_type == "orphan_cluster":
        papers = community_internal_papers(G, candidate.get("members", []))
    else:
        papers = evidence_papers(G, entities)
    specificity = specificity_score(entities)
    provenance = min(len(papers) / max(settings["min_supporting_papers"], 1), 1.0)
    path_diversity = min(
        len(evidence_paths) / max(int(settings.get("min_independent_paths", 2)), 1),
        1.0,
    ) if gap_type == "missing_link" else 1.0

    candidate_for_stability = dict(candidate)
    if evidence_paths:
        candidate_for_stability["independent_evidence_paths"] = evidence_paths
    if gap_type == "temporal_decay" and documents_list:
        counts: defaultdict[int, int] = defaultdict(int)
        for document in documents_list:
            year = document.get("year") or document.get("publication_year")
            try:
                counts[int(year)] += 1
            except (TypeError, ValueError):
                continue
        candidate_for_stability["publication_counts"] = dict(counts)
        snapshot_year = int(str(settings.get("snapshot_date", "0"))[:4] or 0)
        max_year = max(counts, default=0)
        candidate_for_stability["analysis_end_year"] = (
            max_year - 1
            if settings.get("exclude_incomplete_final_year", True) and snapshot_year == max_year
            else max_year
        )
    stability_modes = perturbation_stability(G, candidate_for_stability, settings)
    stability = stability_modes["aggregate"]
    plausible_stress_edge_count = len(_stress_relevant_plausible_edges(candidate_for_stability))
    plausible_stress_available = stability_modes["plausible_edge_addition"] is not None

    existing_edge = bool(
        gap_type == "missing_link"
        and G.has_node(candidate.get("head"))
        and G.has_node(candidate.get("tail"))
        and (G.has_edge(candidate.get("head"), candidate.get("tail"))
             or G.has_edge(candidate.get("tail"), candidate.get("head")))
    )
    closure_available = documents_list is not None
    local_hits = closure_hits(
        candidate,
        documents_list or [],
        token_coverage=float(settings.get("closure_token_coverage", 0.60)),
    )
    closure_clearance = 1.0 if closure_available and not local_hits else 0.0

    weights = settings["weights"]
    metrics = {
        "provenance": round(provenance, 4),
        "specificity": specificity,
        "stability": stability,
        "path_diversity": round(path_diversity, 4),
        "closure_clearance": closure_clearance,
    }
    available_weights = {key: float(value) for key, value in weights.items() if key in metrics}
    score = sum(available_weights[key] * metrics[key] for key in available_weights)
    score /= max(sum(available_weights.values()), 1e-12)

    reasons = []
    canonical_self_link = bool(
        gap_type == "missing_link"
        and canonical_entity_key(candidate.get("head"))
        and canonical_entity_key(candidate.get("head")) == canonical_entity_key(candidate.get("tail"))
    )
    if canonical_self_link:
        reasons.append("canonical_self_link")
    if len(papers) < settings["min_supporting_papers"]:
        reasons.append("insufficient_independent_paper_support")
    if specificity < settings["min_specificity"]:
        reasons.append("generic_or_underspecified_entities")
    if stability < settings["min_stability"] and not existing_edge:
        reasons.append("unstable_under_multi_mode_perturbation")
    if gap_type == "missing_link" and len(evidence_paths) < int(settings.get("min_independent_paths", 2)):
        reasons.append("insufficient_source_disjoint_evidence_paths")
    if existing_edge:
        reasons.append("observed_relation_requires_qualified_review")
    if local_hits:
        reasons.append("possible_prior_coverage_found_in_local_corpus")
    if not closure_available:
        reasons.append("source_closure_corpus_unavailable")
    if not plausible_stress_available:
        reasons.append("plausible_edge_stress_unavailable")

    manual_only_reasons = {
        "observed_relation_requires_qualified_review",
        "possible_prior_coverage_found_in_local_corpus",
        "source_closure_corpus_unavailable",
        "plausible_edge_stress_unavailable",
    }
    if any(reason not in manual_only_reasons for reason in reasons):
        status = "rejected"
    elif existing_edge or local_hits or not closure_available or not plausible_stress_available:
        status = "review_required"
    else:
        status = "automatically_eligible"

    return {
        "status": status,
        "ranking_score": round(score, 4),
        "validation_score": round(score, 4),
        "metrics": metrics,
        "entities": entities,
        "raw_entities": raw_entities,
        "canonical_self_link": canonical_self_link,
        "supporting_paper_count": len(papers),
        "supporting_paper_ids": sorted(papers),
        "independent_evidence_path_count": len(evidence_paths),
        "independent_evidence_paths": evidence_paths,
        "existing_direct_edge": existing_edge,
        "closure_hit_count": len(local_hits),
        "closure_corpus_available": closure_available,
        "closure_hits": local_hits[: settings["max_closure_hits_to_record"]],
        "reasons": reasons,
        "bootstrap": {
            "repeats": settings["bootstrap_repeats"],
            "edge_keep_probability": settings["edge_keep_probability"],
            "paper_keep_probability": settings.get("paper_keep_probability", 0.8),
            "plausible_edge_add_probability": settings.get("plausible_edge_add_probability", 0.5),
            "seed": _stable_seed(candidate, settings["random_seed"]),
            "mode_survival": stability_modes,
            "mode_available": {
                name: value is not None
                for name, value in stability_modes.items()
                if name != "aggregate"
            },
            "plausible_stress_edge_count": plausible_stress_edge_count,
            "plausible_edge_search_performed": bool(
                candidate_for_stability.get("plausible_edge_search_performed") is True
                or candidate_for_stability.get("plausible_edges_evaluated") is True
            ),
        },
    }


def load_corpus_documents(config: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Load a screened corpus from the common filenames used by this project."""
    configured = config.get("gap_validation", {}).get("corpus_path")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    processed = Path(config["paths"]["processed_data"])
    candidates.extend([
        processed / "filtered_papers.json",
        processed / "filtered_corpus.json",
        processed / "screened_papers.json",
    ])
    for path in candidates:
        if path.exists():
            data = load_json(path)
            if isinstance(data, list):
                logger.info("Loaded %d documents for local source closure from %s", len(data), path)
                return data
    logger.warning("No screened corpus found; local source-closure hits will not be computed")
    return None


def validate_all_gaps(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Triage every raw candidate and save evidence-clear candidates plus audit."""
    graph_path = Path(config["paths"]["graph"]) / "knowledge_graph.pkl"
    gaps_path = Path(config["paths"]["outputs"]) / "detected_gaps_raw.json"
    output_dir = ensure_dir(config["paths"]["outputs"])
    if not graph_path.exists() or not gaps_path.exists():
        raise FileNotFoundError("Run the build and detect stages before gap validation")

    with open(graph_path, "rb") as stream:
        G = pickle.load(stream)
    raw_gaps = load_json(gaps_path)
    documents = load_corpus_documents(config)

    eligible = {key: [] for key in ("missing_links", "orphan_clusters", "temporal_decay")}
    review_queue = {key: [] for key in ("missing_links", "orphan_clusters", "temporal_decay")}
    audit_records = []
    for category, candidates in raw_gaps.items():
        for candidate in candidates:
            decision = validate_candidate(G, candidate, config, documents)
            enriched = {**candidate, "validation": decision}
            audit_records.append(enriched)
            if decision["status"] == "automatically_eligible":
                eligible.setdefault(category, []).append(enriched)
            elif decision["status"] == "review_required":
                review_queue.setdefault(category, []).append(enriched)

    summary = {
        "raw_candidates": len(audit_records),
        "automatically_eligible": sum(
            item["validation"]["status"] == "automatically_eligible"
            for item in audit_records
        ),
        "review_required": sum(item["validation"]["status"] == "review_required" for item in audit_records),
        "rejected": sum(item["validation"]["status"] == "rejected" for item in audit_records),
        "decision_rule": "hard evidence contract; ranking_score has no decision threshold",
    }
    save_json(eligible, output_dir / "evidence_clear_candidates.json")
    # Backward-compatible filename for downstream tools; contents use the new
    # automatically_eligible decision label.
    save_json(eligible, output_dir / "validated_gaps.json")
    save_json(review_queue, output_dir / "review_required_gaps.json")
    save_json({"summary": summary, "candidates": audit_records}, output_dir / "gap_validation_audit.json")
    logger.info("Validation gate: %s", summary)
    return eligible


if __name__ == "__main__":
    import yaml

    with open("config.yaml", encoding="utf-8") as stream:
        validate_all_gaps(yaml.safe_load(stream))
