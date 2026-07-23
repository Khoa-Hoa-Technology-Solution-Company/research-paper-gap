"""Deterministic pre-generation screening for isolated-community pairs.

The score is a triage signal, not evidence that two areas should be combined.
It prevents the pipeline from forwarding every possible pair of disconnected
communities to the LLM.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from src import config

STOPWORDS = {
    "a", "an", "and", "analysis", "approach", "article", "articles", "by",
    "data", "for", "from", "in", "information", "is", "literature", "of",
    "directions", "framework", "on", "paper", "papers", "peer-reviewed",
    "research", "review", "reviews", "study", "the",
    "to", "using", "with",
}


def _tokens(nodes: list[str]) -> set[str]:
    text = " ".join(nodes[:config.COMPATIBILITY_TOP_NODES]).lower()
    return {
        token for token in re.findall(r"[a-z][a-z0-9-]{2,}", text)
        if token not in STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@lru_cache(maxsize=1)
def _embedding_model():
    """Load the local sentence encoder once, only when a pair is evaluated."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.SEMANTIC_COMPATIBILITY_MODEL)


def _semantic_similarity(nodes_a: list[str], nodes_b: list[str]) -> tuple[float | None, str]:
    """Compare community descriptors with a pretrained sentence encoder.

    The fallback retains a usable, explicitly labelled lexical screen when the
    model is unavailable (for example, in an offline reproducibility run).
    """
    descriptor_a = "; ".join(nodes_a[:config.COMPATIBILITY_TOP_NODES])
    descriptor_b = "; ".join(nodes_b[:config.COMPATIBILITY_TOP_NODES])
    try:
        model = _embedding_model()
        embeddings = model.encode([descriptor_a, descriptor_b], normalize_embeddings=True)
        similarity = sum(float(a) * float(b) for a, b in zip(embeddings[0], embeddings[1]))
        return similarity, f"sentence-transformers/{config.SEMANTIC_COMPATIBILITY_MODEL}"
    except Exception as exc:
        return None, f"lexical fallback (encoder unavailable: {type(exc).__name__})"


def score_cluster_pair(cluster_a: dict[str, Any], cluster_b: dict[str, Any]) -> dict[str, Any]:
    """Compute a transparent, deterministic compatibility score for a pair."""
    nodes_a = cluster_a.get("representative_nodes", cluster_a["nodes"])
    nodes_b = cluster_b.get("representative_nodes", cluster_b["nodes"])
    tokens_a = _tokens(nodes_a)
    tokens_b = _tokens(nodes_b)
    lexical_overlap = _jaccard(tokens_a, tokens_b)
    semantic_similarity, method = _semantic_similarity(nodes_a, nodes_b)

    # No separate shared-objective feature is inferred here.  Defining one
    # from the same lexical/semantic threshold would double-count semantic
    # evidence; a genuine objective feature needs an ontology or independently
    # validated task classifier, neither of which is available in this pilot.
    semantic_score = semantic_similarity if semantic_similarity is not None else lexical_overlap

    # Low bridge ratios make the structural observation more reliable, but do
    # not themselves establish a research gap.
    structural_evidence = max(
        0.0,
        1.0 - (float(cluster_a["r_bridge"]) + float(cluster_b["r_bridge"])) / 2,
    )
    score = 0.75 * semantic_score + 0.25 * structural_evidence

    return {
        "lexical_overlap": round(lexical_overlap, 4),
        "semantic_similarity": round(semantic_similarity, 4) if semantic_similarity is not None else None,
        "semantic_threshold": config.SEMANTIC_COMPATIBILITY_THRESHOLD,
        "shared_objective": None,
        "shared_objective_status": "not modeled; would require independently validated task/objective labels",
        "technical_compatibility": round(semantic_score, 4),
        "structural_evidence": round(structural_evidence, 4),
        "score": round(score, 4),
        "ranking_score_threshold": config.COMPATIBILITY_THRESHOLD,
        "minimum_content_overlap": config.COMPATIBILITY_MIN_CONTENT_OVERLAP,
        "passed": (
            semantic_score >= config.SEMANTIC_COMPATIBILITY_THRESHOLD
            or lexical_overlap >= config.COMPATIBILITY_MIN_CONTENT_OVERLAP
        ),
        "shared_terms": sorted(tokens_a & tokens_b),
        "representative_node_policy": "top-degree nodes, then lexical tie-break",
        "method": f"{method}; semantic and structural ranking only; score is ranking-only",
    }
