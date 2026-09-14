"""Corroborate structural candidates with gaps that authors state themselves.

Every other signal in this pipeline is *negative*: something is absent from the
graph.  Absence is weak on its own, because it is equally consistent with a
thin corpus or an extractor miss.  This module supplies the one genuinely
*positive* signal available offline: a domain expert, writing in a peer-reviewed
paper, saying in their own words that a line of work is missing.

Two mechanisms are mined:

``lacks_relation``
    A ``LACKS`` edge extracted from an abstract, i.e. the extractor already
    recorded an author asserting that something is missing.
``gap_phrase``
    A sentence in a screened abstract that both matches a curated
    absence-claim pattern and co-mentions the candidate's entities.

Corroboration must be **source-disjoint**.  A statement taken from a paper that
also produced the candidate's own evidence paths proves nothing: that single
paper would be both the premise and the conclusion.  Such statements are
counted and reported separately as circular.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from src.entity_normalization import (
    canonical_entity_key,
    canonical_entity_label,
    entity_tokens,
    mentions_entity,
)
from src.utils import get_logger

logger = get_logger("author_stated_gaps")

# Curated absence-claim patterns.  "explicit" asserts that work does not exist;
# "limitation" only reports a shortcoming, which is weaker evidence of a gap.
GAP_CLAIM_PATTERNS: dict[str, str] = {
    "remains unexplored": "explicit",
    "remain unexplored": "explicit",
    "largely unexplored": "explicit",
    "still unexplored": "explicit",
    "underexplored": "explicit",
    "under-explored": "explicit",
    "understudied": "explicit",
    "under-studied": "explicit",
    "under-researched": "explicit",
    "has not been explored": "explicit",
    "have not been explored": "explicit",
    "has not been investigated": "explicit",
    "have not been investigated": "explicit",
    "has not been studied": "explicit",
    "have not been studied": "explicit",
    "has not been addressed": "explicit",
    "have not been addressed": "explicit",
    "has yet to be": "explicit",
    "have yet to be": "explicit",
    "yet to be explored": "explicit",
    "no prior work": "explicit",
    "no previous work": "explicit",
    "no existing work": "explicit",
    "no existing study": "explicit",
    "no existing studies": "explicit",
    "no studies have": "explicit",
    "few studies have": "explicit",
    "little work has": "explicit",
    "little attention": "explicit",
    "limited attention": "explicit",
    "received little": "explicit",
    "research gap": "explicit",
    "gap in the literature": "explicit",
    "gap in literature": "explicit",
    "knowledge gap": "explicit",
    "open problem": "explicit",
    "open challenge": "explicit",
    "open question": "explicit",
    "remains an open": "explicit",
    "to the best of our knowledge, no": "explicit",
    "to our knowledge, no": "explicit",
    "first to": "limitation",
    "lack of": "limitation",
    "lacks": "limitation",
    "is lacking": "limitation",
    "are lacking": "limitation",
    "warrants further": "limitation",
    "requires further investigation": "limitation",
    "needs further investigation": "limitation",
    "future work should": "limitation",
    "future research should": "limitation",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def matched_claim_patterns(sentence: str) -> list[dict[str, str]]:
    """Return the absence-claim patterns present in one sentence."""
    lowered = sentence.lower()
    return [
        {"phrase": phrase, "tier": tier}
        for phrase, tier in GAP_CLAIM_PATTERNS.items()
        if phrase in lowered
    ]


def candidate_entity_groups(candidate: dict[str, Any], limit: int = 4) -> list[list[str]]:
    """Return the entity groups a statement must co-mention to be relevant.

    Missing links require *both* endpoints, so a statement about one of them
    alone cannot corroborate the pair.
    """
    gap_type = candidate.get("type")
    if gap_type == "missing_link":
        values = [candidate.get("head"), candidate.get("tail")]
    elif gap_type == "temporal_decay":
        values = [candidate.get("concept")]
    else:
        values = list(candidate.get("key_concepts") or candidate.get("members") or [])[:limit]

    aliases = candidate.get("entity_aliases", {}) or {}
    groups: list[list[str]] = []
    for value in values:
        label = canonical_entity_label(value)
        if not label:
            continue
        phrases = [label]
        for alias_entity, alias_values in aliases.items():
            if canonical_entity_key(alias_entity) == canonical_entity_key(label):
                phrases.extend(canonical_entity_label(alias) for alias in alias_values)
        groups.append([phrase for phrase in phrases if phrase])
    return groups


def mine_lacks_relations(
    graph: nx.MultiDiGraph,
    candidate: dict[str, Any],
    entity_groups: list[list[str]],
    coverage: float,
) -> list[dict[str, Any]]:
    """Find LACKS edges in the graph that co-mention candidate entities."""
    statements = []
    for head, tail, data in graph.edges(data=True):
        if data.get("relation_type") != "LACKS":
            continue
        context = " ".join([
            canonical_entity_label(head),
            canonical_entity_label(tail),
            str(data.get("context", ""))
        ])
        context_tokens = entity_tokens(context)
        if not context_tokens:
            continue

        matched_all = True
        for group in entity_groups:
            if not any(
                mentions_entity(context_tokens, entity_tokens(phrase), coverage)
                for phrase in group
            ):
                matched_all = False
                break

        if matched_all:
            statements.append({
                "mechanism": "lacks_relation",
                "head": canonical_entity_label(head),
                "tail": canonical_entity_label(tail),
                "context": str(data.get("context", ""))[:300],
                "source_papers": data.get("source_papers", []),
            })
    return statements


def mine_gap_phrases(
    corpus_documents: list[dict[str, Any]],
    candidate: dict[str, Any],
    entity_groups: list[list[str]],
    coverage: float,
) -> list[dict[str, Any]]:
    """Find sentences that both claim a gap and co-mention candidate entities."""
    statements = []
    for doc in corpus_documents:
        abstract = str(doc.get("abstract", ""))
        paper_id = str(doc.get("paper_id", ""))
        for sentence in _sentences(abstract):
            patterns = matched_claim_patterns(sentence)
            if not patterns:
                continue

            sent_tokens = entity_tokens(sentence)
            if not sent_tokens:
                continue

            matched_all = True
            for group in entity_groups:
                if not any(
                    mentions_entity(sent_tokens, entity_tokens(phrase), coverage)
                    for phrase in group
                ):
                    matched_all = False
                    break

            if matched_all:
                statements.append({
                    "mechanism": "gap_phrase",
                    "sentence": sentence[:500],
                    "patterns": patterns,
                    "source_papers": [paper_id] if paper_id else [],
                })
    return statements


def check_source_disjoint(
    statement_sources: Iterable[str],
    candidate_sources: Iterable[str],
) -> bool:
    """Return whether a statement's sources do not overlap with candidate's."""
    statement_set = {str(src) for src in statement_sources}
    candidate_set = {str(src) for src in candidate_sources}
    return statement_set.isdisjoint(candidate_set)


def corroborate_candidate(
    candidate: dict[str, Any],
    graph: nx.MultiDiGraph,
    corpus_documents: list[dict[str, Any]],
    coverage: float,
) -> dict[str, Any]:
    """Find author-stated gaps that corroborate one structural candidate."""
    entity_groups = candidate_entity_groups(candidate)
    if not entity_groups:
        return {"corroborating": [], "circular": []}

    lacks = mine_lacks_relations(graph, candidate, entity_groups, coverage)
    phrases = mine_gap_phrases(corpus_documents, candidate, entity_groups, coverage)
    all_statements = lacks + phrases

    candidate_sources = set(candidate.get("source_papers", []))
    corroborating = []
    circular = []
    for stmt in all_statements:
        if check_source_disjoint(stmt.get("source_papers", []), candidate_sources):
            corroborating.append(stmt)
        else:
            circular.append(stmt)

    return {
        "corroborating": corroborating,
        "circular": circular,
        "corroboration_count": len(corroborating),
        "circular_count": len(circular),
    }


def corroborate_all_candidates(
    config: dict[str, Any],
    graph: nx.MultiDiGraph | None = None,
    corpus_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage entry point: corroborate every post-gate candidate."""
    from src.validate_gaps import load_kg, load_corpus_documents
    from src.utils import ensure_dir, load_json, save_json

    settings = config.get("author_stated_gaps", {}) or {}
    if not settings.get("enabled", True):
        logger.warning("Author-stated gap mining disabled; skipping")
        return {"enabled": False, "corroborated": 0}

    output_dir = ensure_dir(config["paths"]["outputs"])
    review_path = output_dir / "review_required_gaps.json"
    eligible_path = output_dir / "evidence_clear_candidates.json"
    if not review_path.exists() and not eligible_path.exists():
        raise FileNotFoundError("Run the validate stage before author-stated corroboration")

    if graph is None:
        graph = load_kg(config)
    if corpus_documents is None:
        corpus_documents = load_corpus_documents(config)

    coverage = float(settings.get("co_mention_token_coverage", 0.60))

    queues: list[tuple[str, dict[str, Any]]] = []
    for label, path in (("review_required", review_path), ("automatically_eligible", eligible_path)):
        if not path.exists():
            continue
        for category, candidates in (load_json(path) or {}).items():
            for candidate in candidates:
                queues.append((label, {**candidate, "_queue": label, "_category": category}))

    records = []
    corroborated_count = 0
    for _, candidate in queues:
        result = corroborate_candidate(candidate, graph, corpus_documents, coverage)
        if result["corroboration_count"] > 0:
            corroborated_count += 1
        records.append({
            "type": candidate.get("type"),
            "queue": candidate.get("_queue"),
            "category": candidate.get("_category"),
            "head": candidate.get("head"),
            "tail": candidate.get("tail"),
            "concept": candidate.get("concept"),
            "community_id": candidate.get("community_id"),
            "author_corroboration": result,
        })

    summary = {
        "enabled": True,
        "verified": len(records),
        "corroborated": corroborated_count,
        "policy": "only source-disjoint statements count as corroboration",
    }
    save_json(
        {"summary": summary, "candidates": records},
        output_dir / "author_stated_gaps.json",
    )
    logger.info("Author-stated gap corroboration: %s", summary)
    return summary


# Public alias for pipeline runner
mine_author_stated_gaps = corroborate_all_candidates


if __name__ == "__main__":
    import yaml

    with open("config.yaml", encoding="utf-8") as stream:
        corroborate_all_candidates(yaml.safe_load(stream))
