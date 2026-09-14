"""Synthesize external verification and author corroboration into final rankings.

Every candidate that survived the local validation gate is now enriched with
two additional signals: external index verification (A) and author-stated gap
corroboration (B).  This module applies a policy matrix to combine those
signals, then ranks the survivors by research impact and novelty.

The policy is fail-safe: a refuted candidate is excluded even when author
statements would corroborate it, because a refutation is definitive while a
statement might be outdated or subjective.  A failed probe blocks automatic
approval but does not exclude — human review decides.

Final rankings use the same features that drove the eligibility gate, so a
candidate's position reflects both structural evidence and post-hoc signals.
"""

from __future__ import annotations

from typing import Any

from src.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger("synthesize_rankings")

REFUTED = "refuted"
CORROBORATED = "corroborated"
FAILED = "failed"
NOT_APPLICABLE = "not_applicable"


def _merge_verification_data(
    candidate: dict[str, Any],
    external_record: dict[str, Any] | None,
    author_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine a candidate with its external and author corroboration records."""
    enriched = {**candidate}
    if external_record:
        enriched["external_verification"] = external_record.get("external_verification", {})
    if author_record:
        enriched["author_corroboration"] = author_record.get("author_corroboration", {})
    return enriched


def apply_synthesis_policy(candidate: dict[str, Any]) -> dict[str, str]:
    """Classify a candidate into approved, review_required, or excluded.

    Policy matrix:
    - external=refuted → excluded
    - external=failed → review_required
    - external=corroborated + author≥1 → approved
    - external=corroborated + author=0 → review_required
    - external=not_applicable + author≥1 → approved
    - external=not_applicable + author=0 → review_required
    """
    external = candidate.get("external_verification", {})
    author = candidate.get("author_corroboration", {})

    ext_verdict = external.get("verdict", NOT_APPLICABLE)
    author_count = int(author.get("corroboration_count", 0))

    if ext_verdict == REFUTED:
        return {
            "disposition": "excluded",
            "reason": f"external index refutes absence claim: {len(external.get('refuting_papers', []))} confirming papers",
        }

    if ext_verdict == FAILED:
        return {
            "disposition": "review_required",
            "reason": "external verification probes failed; human judgment required",
        }

    # corroborated or not_applicable
    if author_count > 0:
        return {
            "disposition": "approved",
            "reason": f"external verified + {author_count} author-stated corroboration(s)",
        }

    return {
        "disposition": "review_required",
        "reason": "external verified but no author-stated corroboration found",
    }


def score_candidate(candidate: dict[str, Any]) -> dict[str, float]:
    """Compute ranking scores from structural features and enrichment signals.

    Higher scores indicate stronger research gaps:
    - impact_score: corpus coverage × citation support
    - novelty_score: recency discount × external corroboration boost
    - confidence_score: convergence of all signals
    """
    # Structural features
    corpus_coverage = float(candidate.get("corpus_coverage_fraction", 0.0))
    citations = float(candidate.get("mean_citation_count", 0.0))
    recency = float(candidate.get("publication_recency_score", 0.0))

    # Enrichment signals
    external = candidate.get("external_verification", {})
    author = candidate.get("author_corroboration", {})
    ext_corroborated = external.get("verdict") == CORROBORATED
    author_count = int(author.get("corroboration_count", 0))

    impact_score = corpus_coverage * (1.0 + min(citations / 100.0, 1.0))
    novelty_score = recency * (1.5 if ext_corroborated else 1.0) * (1.0 + 0.2 * min(author_count, 3))
    confidence_score = (
        (1.0 if ext_corroborated else 0.5) * (1.0 + 0.3 * min(author_count, 2))
    )

    return {
        "impact_score": round(impact_score, 4),
        "novelty_score": round(novelty_score, 4),
        "confidence_score": round(confidence_score, 4),
    }


def rank_candidates(
    candidates: list[dict[str, Any]],
    rank_by: str = "confidence_score",
) -> list[dict[str, Any]]:
    """Sort candidates descending by the chosen score, then by impact as tiebreaker."""
    for cand in candidates:
        scores = score_candidate(cand)
        cand.update(scores)

    return sorted(
        candidates,
        key=lambda c: (c.get(rank_by, 0.0), c.get("impact_score", 0.0)),
        reverse=True,
    )


def synthesize_final_rankings(config: dict[str, Any]) -> dict[str, Any]:
    """Stage entry point: merge signals, apply policy, rank, and emit final outputs."""
    settings = config.get("synthesis", {}) or {}
    output_dir = ensure_dir(config["paths"]["outputs"])

    # Load inputs
    review_path = output_dir / "review_required_gaps.json"
    eligible_path = output_dir / "evidence_clear_candidates.json"
    external_path = output_dir / "external_verification.json"
    author_path = output_dir / "author_stated_gaps.json"

    if not review_path.exists() and not eligible_path.exists():
        raise FileNotFoundError("Run the validate stage before synthesis")

    candidates_by_id: dict[str, dict[str, Any]] = {}
    for label, path in (("review_required", review_path), ("automatically_eligible", eligible_path)):
        if not path.exists():
            continue
        for category, candidates in (load_json(path) or {}).items():
            for cand in candidates:
                cand_id = f"{cand.get('type')}:{cand.get('head')}:{cand.get('tail')}:{cand.get('concept')}:{cand.get('community_id')}"
                candidates_by_id[cand_id] = {**cand, "_queue": label, "_category": category}

    external_by_id: dict[str, dict[str, Any]] = {}
    if external_path.exists():
        for rec in (load_json(external_path) or {}).get("candidates", []):
            rec_id = f"{rec.get('type')}:{rec.get('head')}:{rec.get('tail')}:{rec.get('concept')}:{rec.get('community_id')}"
            external_by_id[rec_id] = rec

    author_by_id: dict[str, dict[str, Any]] = {}
    if author_path.exists():
        for rec in (load_json(author_path) or {}).get("candidates", []):
            rec_id = f"{rec.get('type')}:{rec.get('head')}:{rec.get('tail')}:{rec.get('concept')}:{rec.get('community_id')}"
            author_by_id[rec_id] = rec

    # Merge and apply policy
    enriched: list[dict[str, Any]] = []
    for cand_id, cand in candidates_by_id.items():
        merged = _merge_verification_data(
            cand,
            external_by_id.get(cand_id),
            author_by_id.get(cand_id),
        )
        policy = apply_synthesis_policy(merged)
        merged.update(policy)
        enriched.append(merged)

    # Split by disposition
    approved = [c for c in enriched if c.get("disposition") == "approved"]
    review = [c for c in enriched if c.get("disposition") == "review_required"]
    excluded = [c for c in enriched if c.get("disposition") == "excluded"]

    # Rank each partition
    rank_by = str(settings.get("rank_by", "confidence_score"))
    approved = rank_candidates(approved, rank_by)
    review = rank_candidates(review, rank_by)
    excluded = rank_candidates(excluded, rank_by)

    # Assign final ranks
    for rank, cand in enumerate(approved, start=1):
        cand["rank"] = rank
        cand["rank_group"] = "approved"
    for rank, cand in enumerate(review, start=1):
        cand["rank"] = rank
        cand["rank_group"] = "review_required"
    for rank, cand in enumerate(excluded, start=1):
        cand["rank"] = rank
        cand["rank_group"] = "excluded"

    summary = {
        "total": len(enriched),
        "approved": len(approved),
        "review_required": len(review),
        "excluded": len(excluded),
        "rank_by": rank_by,
        "policy": "refuted→excluded; failed→review; corroborated+author≥1→approved",
    }

    save_json(
        {"summary": summary, "approved": approved, "review_required": review, "excluded": excluded},
        output_dir / "final_rankings.json",
    )
    logger.info("Synthesis complete: %s", summary)
    return summary


if __name__ == "__main__":
    import yaml

    with open("config.yaml", encoding="utf-8") as stream:
        synthesize_final_rankings(yaml.safe_load(stream))
