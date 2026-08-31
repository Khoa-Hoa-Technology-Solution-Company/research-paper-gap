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
import time
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from src.entity_normalization import canonical_entity_key, canonical_entity_label
from src.closure_search import search_candidate
from src.utils import ensure_dir, get_logger, load_json, load_jsonl, save_json


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
    "of", "on", "or", "the", "to", "using", "via", "with", "such",
    "their", "under", "that", "these", "this", "those",
}

CLOSURE_ACTION_PATTERN = re.compile(
    r"\b(?:address(?:es|ed|ing)?|mitigat(?:e|es|ed|ing)|solv(?:e|es|ed|ing)|"
    r"prevent(?:s|ed|ing)?|eliminat(?:e|es|ed|ing)|overcom(?:e|es|ing)|"
    r"improv(?:e|es|ed|ing)|outperform(?:s|ed|ing)?|achiev(?:e|es|ed|ing)|"
    r"enabl(?:e|es|ed|ing)|propos(?:e|es|ed|ing)|introduc(?:e|es|ed|ing)|"
    r"develop(?:s|ed|ing)?|implement(?:s|ed|ing)?|integrat(?:e|es|ed|ing)|"
    r"connect(?:s|ed|ing)?|appl(?:y|ies|ied|ying)|secur(?:e|es|ed|ing)|"
    r"design(?:s|ed|ing)?|embed(?:s|ded|ding)?|enhanc(?:e|es|ed|ing)|"
    r"demonstrat(?:e|es|ed|ing)|provid(?:e|es|ed|ing)|"
    r"solution|effective(?:ly)?)\b",
    flags=re.IGNORECASE,
)

# An action word within a limitation clause is not evidence of resolution:
# "traditional methods struggle to address X" describes the gap itself.
NEGATED_RESOLUTION_PATTERN = re.compile(
    r"\b(?:lack(?:s|ed|ing)?|unable|inability|struggl(?:e|es|ed|ing)|"
    r"fail(?:s|ed|ing)?|without|limited|limitation(?:s)?)\b"
    r"(?:\W+\w+){0,4}\W+"
    r"(?:address(?:es|ed|ing)?|mitigat(?:e|es|ed|ing)|solv(?:e|es|ed|ing)|"
    r"prevent(?:s|ed|ing)?|overcom(?:e|es|ing)|detect(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)

_SEMANTIC_ALIASES = {
    "apts": "apt",
    "interpretable": "explainability",
    "interpretability": "explainability",
    "explainable": "explainability",
    "explanations": "explainability",
    "explanation": "explainability",
    "transparent": "explainability",
    "transparency": "explainability",
    "recognizer": "recognition",
    "recognize": "recognition",
    "recognized": "recognition",
    "recognizing": "recognition",
    "detecting": "detect",
    "detected": "detect",
    "detection": "detect",
    "attacks": "attack",
    "threats": "threat",
    "methods": "method",
    "systems": "system",
}

_GENERIC_PROBLEM_TOKENS = {
    "ability", "capability", "existing", "inherent", "lack", "limiting",
    "method", "problem", "system", "traditional",
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
        if token.startswith("secur"):
            output.add("secur")
        if token.startswith("monolith"):
            output.add("monolith")
        if token.startswith("microservic"):
            output.add("microservice")
        if token.endswith("s") and len(token) > 4:
            output.add(token[:-1])
    return output


def _semantic_tokens(text: str) -> set[str]:
    """Normalize common scientific aliases used by deterministic closure gates."""
    tokens = _tokens(text)
    expanded = {_SEMANTIC_ALIASES.get(token, token) for token in tokens}
    raw = canonical_entity_label(text).casefold().replace("-", " ")
    if re.search(r"\bids(?:s)?\b", raw):
        expanded.update({"intrusion", "detect"})
    if re.search(r"\bdl\b", raw):
        expanded.update({"deep", "learning"})
    if re.search(r"\b(?:cnn|rnn|lstm|gan|transformer)s?\b", raw):
        expanded.update({"deep", "learning"})
    if re.search(r"\bml\b", raw):
        expanded.update({"machine", "learning"})
    if re.search(r"\bzero\s+day\b", raw):
        expanded.add("zero_day")
    if re.search(r"\badvanced\s+persistent\s+threat", raw):
        expanded.add("apt")
    return expanded


def _candidate_problem_groups(candidate: dict[str, Any]) -> list[set[str]]:
    """Return independently matchable problem components from a gap candidate.

    Compound claims are intentionally split: prior work resolving even one named
    component is counterevidence and must route the candidate to review.
    """
    missing = str(
        candidate.get("missing_capability")
        or candidate.get("tail")
        or candidate.get("concept")
        or ""
    )
    parts = re.split(r"[,;]|\b(?:and|or)\b", missing, flags=re.IGNORECASE)
    groups = []
    for part in parts:
        tokens = _semantic_tokens(part).difference(_GENERIC_PROBLEM_TOKENS)
        if tokens:
            groups.append(tokens)
    if not groups:
        tokens = _semantic_tokens(missing).difference(_GENERIC_PROBLEM_TOKENS)
        if tokens:
            groups.append(tokens)
    return groups


def _sentence_windows(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or ""))
        if sentence.strip()
    ]
    if not sentences:
        return []
    return [
        " ".join(sentences[max(0, index - 1):min(len(sentences), index + 2)])
        for index in range(len(sentences))
    ]


def document_resolution_matches(
    candidate: dict[str, Any],
    document: dict[str, Any],
    token_coverage: float = 0.60,
) -> list[dict[str, Any]]:
    """Find sentence-local evidence that a paper addresses the claimed problem.

    This is a fail-closed counterevidence screen, not a novelty classifier. A
    matching solution clause blocks automatic acceptance and preserves its text
    for inspection.
    """
    groups = _candidate_problem_groups(candidate)
    if not groups:
        return []
    matches = []
    seen: set[tuple[str, ...]] = set()
    for window in _sentence_windows(_document_text(document)):
        if not CLOSURE_ACTION_PATTERN.search(window):
            continue
        if NEGATED_RESOLUTION_PATTERN.search(window):
            continue
        window_tokens = _semantic_tokens(window)
        matched_groups = []
        for group in groups:
            required = max(1, math.ceil(len(group) * token_coverage))
            overlap = group.intersection(window_tokens)
            # Specific compound anchors such as zero_day, apt, explainability,
            # polymorphic, or regulatory are sufficient counterevidence cues.
            distinctive = {
                token for token in overlap
                if token in {
                    "apt", "explainability", "polymorphic", "regulatory",
                    "zero_day",
                }
            }
            if len(overlap) >= required or distinctive:
                matched_groups.append(tuple(sorted(overlap or distinctive)))
        if not matched_groups:
            continue
        signature = tuple(sorted({token for group in matched_groups for token in group}))
        if signature in seen:
            continue
        seen.add(signature)
        matches.append({
            "matched_problem_tokens": list(signature),
            "evidence": window,
        })
    return matches


def _normalized_doi(value: Any) -> str:
    doi = str(value or "").strip().casefold()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi.rstrip("/.,; ")


def _normalized_title(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _document_identity_keys(document: dict[str, Any]) -> set[str]:
    """Cross-provider identity using IDs, DOI, and normalized title/year."""
    keys: set[str] = set()
    for value in (
        document.get("paperId"),
        document.get("paper_id"),
        document.get("id"),
    ):
        if value:
            keys.add(f"id:{str(value).strip().casefold()}")
    external = document.get("externalIds") or document.get("external_ids") or {}
    doi = _normalized_doi(
        document.get("doi")
        or (external.get("DOI") if isinstance(external, dict) else "")
    )
    if doi:
        keys.add(f"doi:{doi}")
    title = _normalized_title(document.get("title") or document.get("display_name"))
    if title:
        year = document.get("year") or document.get("publication_year") or ""
        keys.add(f"title:{title}")
        if year:
            keys.add(f"title-year:{title}:{year}")
    return keys


def _domain_match(text: str, domain: str) -> tuple[bool, float]:
    """Require substantive lexical coverage of the configured domain."""
    domain_tokens = _tokens(domain)
    if not domain_tokens:
        return True, 1.0
    text_tokens = _tokens(text)
    matched = len(domain_tokens.intersection(text_tokens))
    coverage = matched / max(len(domain_tokens), 1)
    required = 1.0 if len(domain_tokens) <= 2 else 0.6
    return coverage >= required, round(coverage, 4)


_PROBLEM_ANCHOR_FAMILIES = {
    "secur": {
        "secur", "vulnerab", "xss", "attack", "exploit", "threat",
        "isolat", "encrypt", "cryptograph", "authentic", "authoriz",
        "malware", "injection", "breach", "protect", "privacy",
    },
    "recognition": {
        "recogn", "classif", "detect", "accuracy", "symbol",
        "expression", "notation", "transcrib", "parse",
    },
}


def _stemmed_fragments(text: str) -> set[str]:
    """Return token fragments used only for conservative domain alignment."""
    tokens = _tokens(text)
    fragments = set(tokens)
    for token in tokens:
        for length in range(4, min(len(token), 9) + 1):
            fragments.add(token[:length])
    return fragments


def _candidate_problem_match(candidate: dict[str, Any], domain: str) -> tuple[bool, float]:
    """Check that the claimed limitation addresses the domain's problem anchor.

    A source paper can discuss a domain while reporting an unrelated limitation.
    This second check therefore uses only the candidate entities and quoted
    limitation evidence, not the paper's full abstract.
    """
    if not domain:
        return True, 1.0
    evidence = " ".join(
        str(item.get("evidence", ""))
        for item in candidate.get("source_evidence", [])
    )
    candidate_text = " ".join(str(value or "") for value in (
        candidate.get("subject"), candidate.get("missing_capability"),
        candidate.get("head"), candidate.get("tail"), evidence,
    ))
    candidate_fragments = _stemmed_fragments(candidate_text)
    domain_fragments = _stemmed_fragments(domain)
    candidate_semantics = _semantic_tokens(candidate_text)
    domain_semantics = _semantic_tokens(domain)
    lexical_coverage = (
        len(domain_semantics.intersection(candidate_semantics))
        / max(len(domain_semantics), 1)
    )

    applicable = []
    if any(token.startswith("secur") for token in _tokens(domain)):
        applicable.append(_PROBLEM_ANCHOR_FAMILIES["secur"])
    if any(token.startswith("recogn") for token in _tokens(domain)):
        applicable.append(_PROBLEM_ANCHOR_FAMILIES["recognition"])
    if applicable:
        matched = sum(bool(family.intersection(candidate_fragments)) for family in applicable)
        anchor_coverage = matched / len(applicable)
        coverage = max(lexical_coverage, anchor_coverage)
        return matched == len(applicable) or lexical_coverage >= 0.4, round(coverage, 4)

    # Semantic token coverage is interpretable and avoids prefix-fragment counts
    # such as 0.3158 that could pass despite missing the intervention and setting.
    if domain_semantics:
        return lexical_coverage >= 0.4, round(lexical_coverage, 4)
    matched = len(domain_fragments.intersection(candidate_fragments))
    coverage = matched / max(len(domain_fragments), 1)
    return matched >= 1, round(coverage, 4)


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
    elif gap_type == "evidence_gap":
        scope_anchor = (
            candidate.get("domain")
            if candidate.get("semantic_scope") == "field"
            else candidate.get("subject")
        )
        values = [scope_anchor, candidate.get("missing_capability")]
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
    if gap_type == "evidence_gap":
        endpoints = {
            str(candidate.get("subject", "")),
            str(candidate.get("missing_capability", "")),
        }
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


def _explicit_gap_papers(G: nx.Graph, candidate: dict[str, Any]) -> set[str]:
    """Return papers whose surviving graph event explicitly reports ``LACKS``."""
    if candidate.get("evidence_cell_id"):
        records_by_paper: defaultdict[str, set[str]] = defaultdict(set)
        for record in candidate.get("source_evidence", []):
            if record.get("paper_id") and record.get("source_subject"):
                records_by_paper[str(record["paper_id"])].add(
                    str(record["source_subject"])
                )
        surviving = set()
        edges = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
        for edge in edges:
            source, data = str(edge[0]), edge[-1]
            if str(data.get("relation", "")).upper() != "LACKS":
                continue
            paper = str(
                data.get("source_paper") or data.get("source_paper_id") or ""
            )
            if paper and source in records_by_paper.get(paper, set()):
                surviving.add(paper)
        return surviving
    subject = str(candidate.get("subject", ""))
    capability = str(candidate.get("missing_capability", ""))
    papers: set[str] = set()
    pairs = ((subject, capability), (capability, subject)) if G.is_directed() else ((subject, capability),)
    for source, target in pairs:
        if not G.has_edge(source, target):
            continue
        records = (
            G.get_edge_data(source, target, default={}).values()
            if G.is_multigraph()
            else [G.get_edge_data(source, target, default={})]
        )
        for data in records:
            if str(data.get("relation", "")).upper() != "LACKS":
                continue
            paper = data.get("source_paper") or data.get("source_paper_id") or data.get("paper_id")
            if paper:
                papers.add(str(paper))
    return papers


def _has_plausible_closing_edge(G: nx.Graph, candidate: dict[str, Any]) -> bool:
    if candidate.get("evidence_cell_id"):
        # Consolidated cells intentionally do not correspond to one literal KG
        # endpoint pair. Source/local/external closure is evaluated separately
        # against every cell variant and its provenance documents.
        return False
    subject = str(candidate.get("subject", ""))
    capability = str(candidate.get("missing_capability", ""))
    closing_relations = {"PLAUSIBLE", "ADDRESSES", "IMPROVES", "PRODUCES", "EXTENDS"}
    pairs = ((subject, capability), (capability, subject)) if G.is_directed() else ((subject, capability),)
    for source, target in pairs:
        if not G.has_edge(source, target):
            continue
        records = (
            G.get_edge_data(source, target, default={}).values()
            if G.is_multigraph()
            else [G.get_edge_data(source, target, default={})]
        )
        if any(str(data.get("relation", "")).upper() in closing_relations for data in records):
            return True
    return False


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
    if gap_type == "evidence_gap":
        return (
            len(_explicit_gap_papers(graph, candidate))
            >= int(settings.get("min_surviving_explicit_reports", 1))
            and not _has_plausible_closing_edge(graph, candidate)
        )
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
        for key in (
            "title", "abstract", "full_text", "fulltext", "text", "content"
        )
        if document.get(key)
    ).lower()


def _document_identifier(document: dict[str, Any]) -> str:
    return str(
        document.get("paperId")
        or document.get("paper_id")
        or document.get("id")
        or ""
    )


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
        alternatives = [_semantic_tokens(canonical_entity_label(entity))]
        alias_values = []
        for alias_entity, values in aliases.items():
            if canonical_entity_key(alias_entity) == canonical_entity_key(entity):
                alias_values.extend(values)
        alternatives.extend(_semantic_tokens(canonical_entity_label(alias)) for alias in alias_values)
        alternatives = [tokens for tokens in alternatives if tokens]
        if alternatives:
            groups.append(alternatives)
    if not groups:
        return []

    temporal_screen = candidate.get("type") == "temporal_decay" and len(groups) == 1
    domain = str(candidate.get("domain", "")).strip()
    if len(groups) < 2 and not temporal_screen:
        return []

    hits = []
    for index, document in enumerate(documents):
        text = _document_text(document)
        if not text:
            continue
        text_tokens = _semantic_tokens(text)
        domain_relevant, domain_coverage = _domain_match(text, domain)
        resolution_matches = document_resolution_matches(
            candidate, document, token_coverage=token_coverage
        )
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
            is_hit = domain_relevant and bool(resolution_matches)
        if is_hit:
            hits.append({
                "paper_id": document.get("paperId") or document.get("paper_id") or document.get("id") or str(index),
                "title": document.get("title", ""),
                "year": document_year,
                "domain_coverage": domain_coverage,
                "resolution_cue_found": (
                    True if not temporal_screen
                    else None
                ),
                "resolution_matches": resolution_matches[:3],
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
    elif gap_type == "evidence_gap":
        papers = _explicit_gap_papers(G, candidate)
        if not papers:
            papers = {
                str(paper) for paper in candidate.get("supporting_paper_ids", []) if paper
            }
    elif gap_type == "orphan_cluster":
        papers = community_internal_papers(G, candidate.get("members", []))
    else:
        papers = evidence_papers(G, entities)
    specificity = specificity_score(entities)
    required_support = int(
        settings.get("min_explicit_supporting_papers", 1)
        if gap_type == "evidence_gap"
        else settings["min_supporting_papers"]
    )
    provenance = min(len(papers) / max(required_support, 1), 1.0)
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
    closure_documents = [
        document for document in (documents_list or [])
        if _document_identifier(document) not in papers
    ]
    domain = str(candidate.get("domain", "")).strip()
    supporting_documents = [
        document for document in (documents_list or [])
        if _document_identifier(document) in papers
    ]
    source_domain_checks = [
        _domain_match(_document_text(document), domain)
        for document in supporting_documents
    ]
    domain_relevant = (
        any(passed for passed, _ in source_domain_checks)
        if domain and source_domain_checks
        else True
    )
    domain_relevance = (
        max((coverage for _, coverage in source_domain_checks), default=1.0)
        if domain
        else 1.0
    )
    problem_relevant, problem_relevance = _candidate_problem_match(candidate, domain)
    min_problem_relevance = float(settings.get("min_problem_relevance", 0.40))
    problem_relevant = problem_relevant and problem_relevance >= min_problem_relevance
    local_hits = closure_hits(
        candidate,
        closure_documents,
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
        "domain_relevance": round(domain_relevance, 4),
        "candidate_problem_relevance": round(problem_relevance, 4),
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
    if not domain_relevant:
        reasons.append("candidate_out_of_domain")
    if not problem_relevant:
        reasons.append("candidate_problem_relevance_below_threshold")
    if len(papers) < required_support:
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
    if gap_type in {"orphan_cluster", "temporal_decay"}:
        reasons.append("structural_signal_requires_expert_interpretation")

    manual_only_reasons = {
        "observed_relation_requires_qualified_review",
        "possible_prior_coverage_found_in_local_corpus",
        "source_closure_corpus_unavailable",
        "plausible_edge_stress_unavailable",
        "structural_signal_requires_expert_interpretation",
    }
    if any(reason not in manual_only_reasons for reason in reasons):
        status = "rejected"
    elif any(reason in manual_only_reasons for reason in reasons):
        status = "review_required"
    else:
        status = "automatically_eligible"

    if gap_type == "evidence_gap":
        scoped_claim = candidate.get("draft_claim") or (
            f"Within the searched literature, '{candidate.get('missing_capability', '')}' "
            f"remains a reported limitation of '{candidate.get('subject', '')}'."
        )
    elif gap_type == "missing_link":
        scoped_claim = (
            f"Within the searched literature, no direct relation was found between "
            f"'{candidate.get('head', '')}' and '{candidate.get('tail', '')}', despite "
            f"{len(evidence_paths)} independent evidence path(s)."
        )
    else:
        scoped_claim = candidate.get("description", "")

    claim_status = {
        "automatically_eligible": "evidence_cleared_awaiting_expert_review",
        "review_required": "candidate_requires_expert_review",
        "rejected": "not_supported_as_research_gap",
    }[status]

    return {
        "status": status,
        "claim_status": claim_status,
        "scoped_claim": scoped_claim,
        "ranking_score": round(score, 4),
        "validation_score": round(score, 4),
        "metrics": metrics,
        "entities": entities,
        "raw_entities": raw_entities,
        "canonical_self_link": canonical_self_link,
        "domain_relevant": domain_relevant,
        "problem_relevant": problem_relevant,
        "supporting_paper_count": len(papers),
        "supporting_paper_ids": sorted(papers),
        "independent_evidence_path_count": len(evidence_paths),
        "independent_evidence_paths": evidence_paths,
        "existing_direct_edge": existing_edge,
        "closure_hit_count": len(local_hits),
        "closure_corpus_available": closure_available,
        "closure_hits": local_hits[: settings["max_closure_hits_to_record"]],
        "external_closure_search": candidate.get("external_closure_search", {}),
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
        processed / "corpus_filtered.jsonl",
        processed / "filtered_papers.json",
        processed / "filtered_corpus.json",
        processed / "screened_papers.json",
    ])
    for path in candidates:
        if path.exists():
            data = load_jsonl(path) if path.suffix == ".jsonl" else load_json(path)
            if isinstance(data, list):
                logger.info("Loaded %d documents for local source closure from %s", len(data), path)
                return data
    logger.warning("No screened corpus found; local source-closure hits will not be computed")
    return None


def _candidate_audit_key(candidate: dict[str, Any]) -> str:
    payload = {
        "type": candidate.get("type"),
        "subject": candidate.get("subject"),
        "missing_capability": candidate.get("missing_capability"),
        "head": candidate.get("head"),
        "tail": candidate.get("tail"),
        "concept": candidate.get("concept"),
        "members": sorted(map(str, candidate.get("members", []))),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"{candidate.get('type', 'candidate')}_{digest}"


def _plausible_edges_from_closure_hits(
    candidate: dict[str, Any],
    hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if candidate.get("type") == "evidence_gap":
        head, tail = candidate.get("subject"), candidate.get("missing_capability")
    elif candidate.get("type") == "missing_link":
        head, tail = candidate.get("head"), candidate.get("tail")
    else:
        return []
    return [
        {
            "head": str(head),
            "tail": str(tail),
            "relation": "PLAUSIBLE",
            "paper_id": hit.get("paper_id"),
            "year": hit.get("year"),
            "title": hit.get("title", ""),
            "confidence": 1.0,
        }
        for hit in hits
        if head and tail and hit.get("paper_id")
    ]


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

    categories = ("evidence_gaps", "missing_links", "orphan_clusters", "temporal_decay")
    eligible = {key: [] for key in categories}
    review_queue = {key: [] for key in categories}
    audit_records = []
    closure_path = output_dir / "gap_closure_search.json"
    closure_report = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "records": {},
    }
    if closure_path.exists():
        try:
            existing_report = load_json(closure_path)
            if isinstance(existing_report, dict):
                closure_report["records"] = existing_report.get("records", {})
        except (OSError, ValueError, TypeError):
            pass

    external_settings = config.get("gap_validation", {}).get("external_closure", {})
    external_enabled = bool(external_settings.get("enabled", False))
    synthesis_minimum = int(
        config.get("gap_synthesis", {}).get("min_screened_corpus_size", 0)
    )
    screened_corpus_size = len(documents or [])
    external_pause_reason = None
    if external_enabled and screened_corpus_size < synthesis_minimum:
        external_pause_reason = (
            "screened_corpus_below_synthesis_minimum: "
            f"{screened_corpus_size} < {synthesis_minimum}"
        )
        logger.warning(
            "Skipping external closure search because %s. The synthesis stage "
            "will record a fail-closed null result.",
            external_pause_reason,
        )
    external_budget = max(int(external_settings.get("max_candidates", 15)), 0)
    external_delay = max(float(external_settings.get("delay_between_queries", 1.0)), 0.0)
    external_attempts = 0
    external_completed = 0
    external_skipped_preflight = 0
    external_skipped_domain_preflight = 0
    consecutive_external_failures = 0
    rate_limit_circuit_threshold = max(
        int(external_settings.get(
            "failure_circuit_breaker",
            external_settings.get("rate_limit_circuit_breaker", 2),
        )), 1
    )
    category_order = [key for key in categories if key in raw_gaps]
    category_order.extend(key for key in raw_gaps if key not in category_order)

    for category in category_order:
        for candidate in raw_gaps.get(category, []):
            candidate_for_validation = dict(candidate)
            candidate_for_validation.setdefault(
                "domain", config.get("project", {}).get("domain", "")
            )
            candidate_key = _candidate_audit_key(candidate)
            candidate_documents = list(documents or [])
            domain = str(candidate_for_validation.get("domain", ""))
            problem_ready, problem_coverage = _candidate_problem_match(
                candidate_for_validation, domain
            )
            problem_ready = problem_ready and problem_coverage >= float(
                config.get("gap_validation", {}).get("min_problem_relevance", 0.40)
            )
            supporting_ids = {
                str(value)
                for value in candidate.get("supporting_paper_ids", [])
                if value
            }
            supporting_documents = [
                paper for paper in (documents or [])
                if _document_identifier(paper) in supporting_ids
            ]
            source_domain_ready = (
                any(_domain_match(_document_text(paper), domain)[0]
                    for paper in supporting_documents)
                if domain and supporting_documents else True
            )
            domain_preflight_ready = problem_ready and source_domain_ready
            external_candidate_ready = (
                candidate.get("type") == "evidence_gap"
                and domain_preflight_ready
                and bool(
                    candidate.get("candidate_quality", {})
                    .get("certificate_readiness", {})
                    .get("independent_sources_ready", False)
                )
            )
            if candidate.get("type") == "missing_link":
                preflight_paths = independent_evidence_paths(
                    G,
                    str(candidate.get("head", "")),
                    str(candidate.get("tail", "")),
                    cutoff=int(
                        config.get("gap_validation", {}).get("max_path_length", 4)
                    ),
                )
                external_candidate_ready = len(preflight_paths) >= int(
                    config.get("gap_validation", {}).get("min_independent_paths", 2)
                )
            should_search = (
                external_enabled
                and external_pause_reason is None
                and external_candidate_ready
                and external_attempts < external_budget
            )
            if (
                external_enabled
                and candidate.get("type") == "missing_link"
                and not external_candidate_ready
            ):
                external_skipped_preflight += 1
            elif (
                external_enabled
                and candidate.get("type") == "evidence_gap"
                and not domain_preflight_ready
            ):
                external_skipped_preflight += 1
                external_skipped_domain_preflight += 1
            closure_record = closure_report["records"].get(candidate_key)
            if should_search and not (
                isinstance(closure_record, dict) and closure_record.get("performed") is True
            ):
                closure_record = search_candidate(candidate, config)
                closure_report["records"][candidate_key] = closure_record
                closure_report["updated_at"] = datetime.now(timezone.utc).isoformat()
                save_json(closure_report, closure_path)
                external_attempts += 1
                if closure_record.get("performed"):
                    external_completed += 1
                    consecutive_external_failures = 0
                else:
                    consecutive_external_failures += 1
                    if consecutive_external_failures >= rate_limit_circuit_threshold:
                        failure_text = str(
                            closure_record.get("error", "")
                        ).casefold()
                        failure_kind = (
                            "rate_limit" if "rate-limit" in failure_text
                            else "provider_unavailable"
                        )
                        external_pause_reason = (
                            f"external_closure_{failure_kind}_circuit_open_after_"
                            f"{consecutive_external_failures}_candidate_failures"
                        )
                        logger.warning(
                            "Pausing remaining closure searches: %s",
                            external_pause_reason,
                        )
                if external_delay and external_attempts < external_budget:
                    time.sleep(external_delay)
            elif should_search:
                external_attempts += 1
                external_completed += 1

            if isinstance(closure_record, dict):
                external_papers = list(closure_record.get("papers", []))
                supporting_ids = {
                    str(value) for value in candidate.get("supporting_paper_ids", []) if value
                }
                source_identity_keys: set[str] = set()
                for paper in documents or []:
                    if _document_identifier(paper) in supporting_ids:
                        source_identity_keys.update(_document_identity_keys(paper))
                searchable_papers = [
                    paper for paper in external_papers
                    if _document_identifier(paper) not in supporting_ids
                    and not (
                        source_identity_keys
                        and _document_identity_keys(paper).intersection(
                            source_identity_keys
                        )
                    )
                ]
                candidate_documents.extend(searchable_papers)
                external_hits = closure_hits(
                    candidate_for_validation,
                    searchable_papers,
                    token_coverage=float(
                        config.get("gap_validation", {}).get("closure_token_coverage", 0.60)
                    ),
                )
                closure_record_for_candidate = {
                    key: value for key, value in closure_record.items() if key != "papers"
                }
                closure_record_for_candidate["closure_hits"] = external_hits
                closure_record_for_candidate["retrieved_paper_ids"] = [
                    _document_identifier(paper) for paper in external_papers
                    if _document_identifier(paper)
                ]
                candidate_for_validation["external_closure_search"] = closure_record_for_candidate
                candidate_for_validation["plausible_edge_search_performed"] = bool(
                    closure_record.get("performed")
                )
                candidate_for_validation["plausible_edges_evaluated"] = bool(
                    closure_record.get("performed")
                )
                candidate_for_validation["plausible_edges"] = _plausible_edges_from_closure_hits(
                    candidate,
                    external_hits,
                )

            decision = validate_candidate(
                G,
                candidate_for_validation,
                config,
                candidate_documents if documents is not None or closure_record else None,
            )
            enriched = {**candidate_for_validation, "validation": decision}
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
        "external_closure_searches_completed": external_completed,
        "external_closure_search_budget": external_budget,
        "external_closure_searches_attempted": external_attempts,
        "external_closure_candidates_skipped_preflight": external_skipped_preflight,
        "external_closure_candidates_skipped_domain_preflight": (
            external_skipped_domain_preflight
        ),
        "external_closure_pause_reason": external_pause_reason,
        "screened_corpus_size": screened_corpus_size,
        "synthesis_minimum_corpus_size": synthesis_minimum,
        "decision_rule": "hard evidence contract; ranking_score has no decision threshold",
    }
    save_json(eligible, output_dir / "evidence_clear_candidates.json")
    # Backward-compatible filename for downstream tools; contents use the new
    # automatically_eligible decision label.
    save_json(eligible, output_dir / "validated_gaps.json")
    save_json(review_queue, output_dir / "review_required_gaps.json")
    save_json({"summary": summary, "candidates": audit_records}, output_dir / "gap_validation_audit.json")
    evidence_claims = []
    for category in categories:
        for candidate in eligible.get(category, []):
            evidence_claims.append({
                "claim_id": _candidate_audit_key(candidate),
                "claim": candidate["validation"]["scoped_claim"],
                "claim_status": candidate["validation"]["claim_status"],
                "candidate_type": candidate.get("type"),
                "supporting_paper_ids": candidate["validation"].get(
                    "supporting_paper_ids", []
                ),
                "closure_hits": candidate["validation"].get("closure_hits", []),
                "validation": candidate["validation"],
                "candidate": candidate,
            })
    save_json(evidence_claims, output_dir / "research_gap_claims.json")
    logger.info("Validation gate: %s", summary)
    return eligible


if __name__ == "__main__":
    import yaml

    with open("config.yaml", encoding="utf-8") as stream:
        validate_all_gaps(yaml.safe_load(stream))
