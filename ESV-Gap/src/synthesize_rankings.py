"""Synthesize external verification, author corroboration, and self-assessments.

Operationalizes the Four-Pillar Evidence Triage policy:
Pillar A: External counterevidence acquisition
Pillar B: Source-disjoint author corroboration
Pillar C: Extractor reliability self-assessment
Pillar D: Corpus coverage self-assessment

The policy is fail-closed:
- A refuted candidate is excluded/refuted even if author statements exist.
- A failed external probe blocks support and routes to review_required.
- An uncorroborated candidate routes to review_required.
- Corroborated candidates in an UNSATURATED corpus are categorized as
  'evidence_supported_but_open' rather than claiming absolute global absence.
"""

from __future__ import annotations

from typing import Any

from src.autonomous_reasoning import (
    AutonomousAction,
    EpistemicDisposition,
    decide_hypothesis,
)
from src.external_verification import (
    CORROBORATED as EXT_CORROBORATED,
    FAILED as EXT_FAILED,
    NOT_APPLICABLE as EXT_NOT_APPLICABLE,
    REFUTED as EXT_REFUTED,
)
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


def apply_synthesis_policy(
    candidate: dict[str, Any],
    saturation_summary: dict[str, Any] | None = None,
    extractor_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a candidate into refuted, evidence_supported, evidence_supported_but_open, or review_required.

    Non-compensatory policy matrix:
    - external=refuted -> refuted (excluded)
    - external=failed -> review_required
    - external=corroborated + author>=1 + corpus=SATURATED -> evidence_supported
    - external=corroborated + author>=1 + corpus!=SATURATED -> evidence_supported_but_open
    - external=corroborated + author=0 -> review_required
    - external=not_applicable -> review_required
    """
    external = candidate.get("external_verification", {})
    author = candidate.get("author_corroboration", {})

    decision = decide_hypothesis(
        candidate=candidate,
        external_record=external,
        author_record=author,
        saturation_summary=saturation_summary,
        extractor_summary=extractor_summary,
    )

    return {
        "disposition": decision.epistemic_disposition.value,
        "epistemic_disposition": decision.epistemic_disposition.value,
        "action": decision.action.value,
        "reason": decision.decision_reason,
        "decision_reason": decision.decision_reason,
        "abstention_reason": decision.abstention_reason,
        "decision_object": decision.to_dict(),
    }


def score_candidate(candidate: dict[str, Any]) -> dict[str, float]:
    """Compute ranking scores from structural features and enrichment signals.

    Note: ranking scores are heuristic and strictly separate from epistemic gates.
    A high score can never overturn a failed gate or refute counterevidence.
    """
    corpus_coverage = float(candidate.get("corpus_coverage_fraction", 0.0))
    citations = float(candidate.get("mean_citation_count", 0.0))
    recency = float(candidate.get("publication_recency_score", 0.0))

    external = candidate.get("external_verification", {})
    author = candidate.get("author_corroboration", {})
    ext_verdict = external.get("verdict")
    ext_corroborated = ext_verdict in (CORROBORATED, EXT_CORROBORATED)
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
    """Stage entry point: merge signals, apply Four-Pillar policy, rank, and emit outputs."""
    from src.gap_provenance import candidate_identity

    settings = config.get("synthesis", {}) or {}
    output_dir = ensure_dir(config["paths"]["outputs"])

    # Load inputs
    review_path = output_dir / "review_required_gaps.json"
    eligible_path = output_dir / "evidence_clear_candidates.json"
    external_path = output_dir / "external_verification.json"
    author_path = output_dir / "author_stated_gaps.json"
    recall_path = output_dir / "extractor_recall_report.json"
    saturation_path = output_dir / "corpus_saturation_report.json"

    if not review_path.exists() and not eligible_path.exists():
        raise FileNotFoundError("Run the validate stage before synthesis")

    # Load Pillar C & D data first so policy can consume them
    extractor_summary: dict[str, Any] = {}
    if recall_path.exists():
        recall_data = load_json(recall_path) or {}
        extractor_summary = recall_data.get("summary", {})

    saturation_summary: dict[str, Any] = {}
    if saturation_path.exists():
        sat_data = load_json(saturation_path) or {}
        saturation_summary = sat_data.get("summary", {})

    candidates_by_id: dict[str, dict[str, Any]] = {}
    for label, path in (("review_required", review_path), ("automatically_eligible", eligible_path)):
        if not path.exists():
            continue
        for category, candidates in (load_json(path) or {}).items():
            for cand in candidates:
                cand_id = candidate_identity(cand)
                candidates_by_id[cand_id] = {**cand, "_queue": label, "_category": category}

    external_by_id: dict[str, dict[str, Any]] = {}
    if external_path.exists():
        for rec in (load_json(external_path) or {}).get("candidates", []):
            rec_id = candidate_identity(rec)
            external_by_id[rec_id] = rec

    author_by_id: dict[str, dict[str, Any]] = {}
    if author_path.exists():
        for rec in (load_json(author_path) or {}).get("candidates", []):
            rec_id = candidate_identity(rec)
            author_by_id[rec_id] = rec

    # Merge and apply calibrated policy
    enriched: list[dict[str, Any]] = []
    for cand_id, cand in candidates_by_id.items():
        merged = _merge_verification_data(
            cand,
            external_by_id.get(cand_id),
            author_by_id.get(cand_id),
        )
        policy = apply_synthesis_policy(
            merged,
            saturation_summary=saturation_summary,
            extractor_summary=extractor_summary,
        )
        merged.update(policy)
        enriched.append(merged)

    # Split by disposition
    evidence_supported = [
        c for c in enriched
        if c.get("epistemic_disposition") == EpistemicDisposition.EVIDENCE_SUPPORTED.value
    ]
    evidence_supported_but_open = [
        c for c in enriched
        if c.get("epistemic_disposition") == EpistemicDisposition.EVIDENCE_SUPPORTED_BUT_OPEN.value
    ]
    review = [
        c for c in enriched
        if c.get("epistemic_disposition") == EpistemicDisposition.REVIEW_REQUIRED.value
    ]
    refuted = [
        c for c in enriched
        if c.get("epistemic_disposition") == EpistemicDisposition.REFUTED.value
    ]

    # Rank each partition
    rank_by = str(settings.get("rank_by", "confidence_score"))
    evidence_supported = rank_candidates(evidence_supported, rank_by)
    evidence_supported_but_open = rank_candidates(evidence_supported_but_open, rank_by)
    review = rank_candidates(review, rank_by)
    refuted = rank_candidates(refuted, rank_by)

    # Assign ranks
    all_approved = evidence_supported + evidence_supported_but_open
    for rank, cand in enumerate(all_approved, start=1):
        cand["rank"] = rank
        cand["rank_group"] = cand["epistemic_disposition"]
    for rank, cand in enumerate(review, start=1):
        cand["rank"] = rank
        cand["rank_group"] = "review_required"
    for rank, cand in enumerate(refuted, start=1):
        cand["rank"] = rank
        cand["rank_group"] = "refuted"

    pillars: dict[str, Any] = {
        "A_external_verification": {"enabled": external_path.exists(), "probed_candidates": len(external_by_id)},
        "B_author_stated_gaps": {"enabled": author_path.exists(), "mined_candidates": len(author_by_id)},
        "C_extractor_recall": extractor_summary,
        "D_corpus_saturation": saturation_summary,
    }

    summary = {
        "total": len(enriched),
        "approved": len(all_approved),
        "evidence_supported": len(evidence_supported),
        "evidence_supported_but_open": len(evidence_supported_but_open),
        "review_required": len(review),
        "excluded": len(refuted),
        "refuted": len(refuted),
        "rank_by": rank_by,
        "policy": (
            "refuted->excluded; failed->review_required; "
            "corroborated+author>=1+SATURATED->evidence_supported; "
            "corroborated+author>=1+UNSATURATED->evidence_supported_but_open"
        ),
        "evidence_pillars": pillars,
    }

    save_json(
        {
            "summary": summary,
            "approved": all_approved,
            "evidence_supported": evidence_supported,
            "evidence_supported_but_open": evidence_supported_but_open,
            "review_required": review,
            "excluded": refuted,
            "refuted": refuted,
        },
        output_dir / "final_rankings.json",
    )
    logger.info("Synthesis complete: %s", summary)
    return summary


if __name__ == "__main__":
    import yaml

    with open("config.yaml", encoding="utf-8") as stream:
        synthesize_final_rankings(yaml.safe_load(stream))
