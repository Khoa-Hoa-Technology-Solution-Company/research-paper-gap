"""Autonomous Scientific Reasoning Engine for Evidence-Grounded Hypothesis Triage.

Operationalizes the closed-loop autonomous reasoning architecture:
OBSERVE -> HYPOTHESIZE -> VERIFY -> REASON -> DECIDE -> ACT / ABSTAIN

This module formalizes:
1. Graded epistemic dispositions:
   - REFUTED: Confirmed external counterevidence found.
   - EVIDENCE_SUPPORTED: External absence corroborated, independent corroboration exists,
     and corpus vocabulary is SATURATED.
   - EVIDENCE_SUPPORTED_BUT_OPEN: External absence corroborated, independent corroboration exists,
     but corpus diagnostics indicate an UNSATURATED or MATURING knowledge space.
   - REVIEW_REQUIRED: External probe failed, insufficient corroboration, or critical ambiguity.
2. Structured decision objects:
   - VerificationResult: Auditable probe outcomes with query provenance and inspection counts.
   - HypothesisDecision: Complete epistemic decision record with multi-pillar provenance.
3. Fail-closed decision policy:
   - External errors fail closed to REVIEW_REQUIRED.
   - Pure topological absence (missing KG edge alone) never yields a supported disposition.
   - Heuristic ranking scores cannot override failed evidence gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable


class EpistemicDisposition(str, Enum):
    """Calibrated epistemic status of a scientific hypothesis."""

    REFUTED = "refuted"
    EVIDENCE_SUPPORTED = "evidence_supported"
    EVIDENCE_SUPPORTED_BUT_OPEN = "evidence_supported_but_open"
    REVIEW_REQUIRED = "review_required"


class AutonomousAction(str, Enum):
    """Downstream epistemic action executed by the autonomous agent."""

    REJECT = "reject_hypothesis"
    RETAIN_AND_SYNTHESIZE = "retain_and_synthesize_pmcost"
    RETAIN_WITH_WARNING = "retain_with_epistemic_warning_and_synthesize"
    ABSTAIN_AND_ESCALATE = "abstain_and_escalate_to_human_review"


@dataclass
class VerificationResult:
    """Auditable query-level external evidence probe result."""

    provider: str
    query: str
    query_status: str
    raw_count: int = 0
    inspected_count: int = 0
    confirmed_counterevidence_count: int = 0
    timestamp: str = ""
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisDecision:
    """Complete auditable decision record produced by the epistemic policy."""

    candidate_id: str
    subject: str
    missing_capability: str
    local_validation_status: str
    external_verification_verdict: str
    counterevidence_count: int
    independent_corroboration_count: int
    circular_corroboration_count: int
    corpus_coverage_status: str
    extractor_reliability: dict[str, Any]
    epistemic_disposition: EpistemicDisposition
    action: AutonomousAction
    decision_reason: str
    abstention_reason: str | None
    timestamp: str
    scores: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["epistemic_disposition"] = self.epistemic_disposition.value
        data["action"] = self.action.value
        return data


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def decide_hypothesis(
    candidate: dict[str, Any],
    external_record: dict[str, Any] | None = None,
    author_record: dict[str, Any] | None = None,
    saturation_summary: dict[str, Any] | None = None,
    extractor_summary: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> HypothesisDecision:
    """Assign a graded epistemic disposition and action to a candidate hypothesis.

    Non-compensatory policy:
    1. External counterevidence found -> REFUTED (Action: REJECT)
    2. External verification failed / rate-limited -> REVIEW_REQUIRED (Action: ABSTAIN_AND_ESCALATE)
    3. External absence corroborated:
       - If independent author corroboration < 1 -> REVIEW_REQUIRED (insufficient independent evidence)
       - If independent author corroboration >= 1:
           - If corpus is SATURATED -> EVIDENCE_SUPPORTED
           - If corpus is UNSATURATED or MATURING -> EVIDENCE_SUPPORTED_BUT_OPEN
    4. Topological absence alone (no external check or unverified type) -> REVIEW_REQUIRED
    """
    from src.gap_provenance import candidate_identity

    ts = timestamp or _utc_now()
    cand_id = candidate.get("evidence_cell_id") or candidate_identity(candidate)
    subject = str(candidate.get("subject") or candidate.get("head") or "")
    missing_cap = str(candidate.get("missing_capability") or candidate.get("tail") or candidate.get("concept") or "")
    local_status = str(candidate.get("_queue") or candidate.get("local_status") or "review_required")

    # Extract Pillar A (External Verification)
    ext_data = {}
    if external_record:
        ext_data = external_record.get("external_verification", external_record)
    elif "external_verification" in candidate:
        ext_data = candidate["external_verification"]

    ext_verdict = str(ext_data.get("verdict", "not_applicable")).strip().lower()
    refuting_papers = ext_data.get("refuting_papers", []) or []
    counterevidence_count = len(refuting_papers) or int(ext_data.get("confirmed_hit_count", 0))

    # Extract Pillar B (Author Corroboration)
    auth_data = {}
    if author_record:
        auth_data = author_record.get("author_corroboration", author_record)
    elif "author_corroboration" in candidate:
        auth_data = candidate["author_corroboration"]

    independent_corrob_count = int(auth_data.get("corroboration_count", 0))
    circular_count = int(auth_data.get("circular_count", 0))
    supp_ids = candidate.get("supporting_paper_ids") or []
    if not supp_ids and author_record:
        supp_ids = [
            pid
            for item in (auth_data.get("corroborating", []) + auth_data.get("circular", []))
            for pid in item.get("source_papers", [])
        ]
    supporting_paper_count = int(candidate.get("supporting_paper_count") or len(supp_ids) or (1 if independent_corrob_count > 0 else 0))

    # Extract Pillar C (Extractor Reliability)
    ext_rel = {}
    if extractor_summary:
        ext_rel = {
            "recall": extractor_summary.get("recall"),
            "miss_rate": extractor_summary.get("miss_rate"),
            "precision": extractor_summary.get("precision"),
            "f1_score": extractor_summary.get("f1_score"),
            "risk_interpretation": (
                "Empirical estimate of extraction-related false-absence risk. "
                "Graph absence cannot independently establish scientific novelty."
            ),
        }
    else:
        ext_rel = {
            "recall": candidate.get("extractor_recall"),
            "miss_rate": candidate.get("extractor_miss_rate"),
            "risk_interpretation": "Extraction reliability unquantified.",
        }

    # Extract Pillar D (Corpus Coverage / Saturation)
    sat_summary = saturation_summary or candidate.get("corpus_saturation", {})
    corpus_status = str(sat_summary.get("verdict", "UNSATURATED")).strip().upper()

    # Apply Fail-Closed Non-Compensatory Epistemic Rules
    if ext_verdict in ("refuted_by_external_literature", "refuted") or counterevidence_count > 0:
        disposition = EpistemicDisposition.REFUTED
        action = AutonomousAction.REJECT
        decision_reason = (
            f"External literature search identified {counterevidence_count} confirmed "
            "counterevidence publications addressing this problem."
        )
        abstention_reason = None

    elif ext_verdict in ("verification_failed", "failed"):
        disposition = EpistemicDisposition.REVIEW_REQUIRED
        action = AutonomousAction.ABSTAIN_AND_ESCALATE
        decision_reason = "External verification probes failed or were rate-limited; fail-closed policy applies."
        abstention_reason = "External index probes failed or uncompleted; absence unverified."

    elif ext_verdict in ("not_applicable", ""):
        disposition = EpistemicDisposition.REVIEW_REQUIRED
        action = AutonomousAction.ABSTAIN_AND_ESCALATE
        decision_reason = "No external absence probe defined for this candidate type; graph absence alone is insufficient."
        abstention_reason = "Topological anomaly lacks external evidence grounding."

    elif ext_verdict in ("absence_corroborated", "corroborated"):
        # Candidate has zero confirmed counterevidence hits under executed queries
        if independent_corrob_count < 1:
            disposition = EpistemicDisposition.REVIEW_REQUIRED
            action = AutonomousAction.ABSTAIN_AND_ESCALATE
            decision_reason = (
                "External queries identified no confirmed counterevidence, but independent "
                "author corroboration is lacking (0 source-disjoint corroborating statements)."
            )
            abstention_reason = "Absence of external hits is uncorroborated by independent peer literature."
        elif supporting_paper_count < 1:
            disposition = EpistemicDisposition.REVIEW_REQUIRED
            action = AutonomousAction.ABSTAIN_AND_ESCALATE
            decision_reason = "Candidate lacks local evidence provenance (0 supporting papers)."
            abstention_reason = "Candidate lacks verifiable local evidence provenance."
        else:
            # External absence corroborated AND independent corroboration exists
            if corpus_status == "SATURATED":
                disposition = EpistemicDisposition.EVIDENCE_SUPPORTED
                action = AutonomousAction.RETAIN_AND_SYNTHESIZE
                decision_reason = (
                    f"External absence corroborated under executed queries; {independent_corrob_count} "
                    "independent source-disjoint corroboration(s); corpus terminology SATURATED."
                )
                abstention_reason = None
            else:
                # UNSATURATED or MATURING
                disposition = EpistemicDisposition.EVIDENCE_SUPPORTED_BUT_OPEN
                action = AutonomousAction.RETAIN_WITH_WARNING
                decision_reason = (
                    f"External absence corroborated under executed queries; {independent_corrob_count} "
                    f"independent source-disjoint corroboration(s); but corpus coverage is {corpus_status} "
                    f"(Heaps beta={sat_summary.get('heaps_law', {}).get('beta', 'N/A')}). "
                    "Candidate is evidence-supported within an actively expanding vocabulary space."
                )
                abstention_reason = None
    else:
        disposition = EpistemicDisposition.REVIEW_REQUIRED
        action = AutonomousAction.ABSTAIN_AND_ESCALATE
        decision_reason = f"Unknown external verification verdict: {ext_verdict}."
        abstention_reason = f"Unrecognized external verdict state '{ext_verdict}'."

    scores = {
        "impact_score": float(candidate.get("impact_score", 0.0)),
        "novelty_score": float(candidate.get("novelty_score", 0.0)),
        "confidence_score": float(candidate.get("confidence_score", 0.0)),
    }

    provenance = {
        "supporting_paper_ids": candidate.get("supporting_paper_ids", []),
        "corroborating_sources": [
            s.get("source_papers", [])
            for s in auth_data.get("corroborating", [])
            if isinstance(s, dict)
        ],
        "refuting_papers": refuting_papers,
        "external_probes": ext_data.get("probes", []),
    }

    return HypothesisDecision(
        candidate_id=cand_id,
        subject=subject,
        missing_capability=missing_cap,
        local_validation_status=local_status,
        external_verification_verdict=ext_verdict,
        counterevidence_count=counterevidence_count,
        independent_corroboration_count=independent_corrob_count,
        circular_corroboration_count=circular_count,
        corpus_coverage_status=corpus_status,
        extractor_reliability=ext_rel,
        epistemic_disposition=disposition,
        action=action,
        decision_reason=decision_reason,
        abstention_reason=abstention_reason,
        timestamp=ts,
        scores=scores,
        provenance=provenance,
    )
