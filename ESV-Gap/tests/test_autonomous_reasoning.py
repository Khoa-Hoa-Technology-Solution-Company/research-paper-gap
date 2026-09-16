"""Tests for Autonomous Scientific Reasoning and Calibrated Epistemic Decisions.

Verifies the 9 required AMI behaviors:
TEST 1: External counterevidence found -> REFUTED regardless of ranking score.
TEST 2: External verification successful, zero confirmed counterevidence,
        independent corroboration exists, corpus SATURATED -> EVIDENCE_SUPPORTED.
TEST 3: External verification successful, zero confirmed counterevidence,
        independent corroboration exists, corpus UNSATURATED -> EVIDENCE_SUPPORTED_BUT_OPEN.
TEST 4: External verification API failure -> REVIEW_REQUIRED.
TEST 5: Corroborating evidence from same source -> must NOT increment independent corroboration.
TEST 6: Missing KG edge alone -> must NOT produce a supported disposition.
TEST 7: Extractor miss rate -> affects reliability metadata, not P(true_gap | absent_edge).
TEST 8: Decision provenance is preserved.
TEST 9: Frozen experiment reproduces candidate statuses under calibrated terminology.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.author_stated_gaps import check_source_disjoint
from src.autonomous_reasoning import (
    AutonomousAction,
    EpistemicDisposition,
    HypothesisDecision,
    VerificationResult,
    decide_hypothesis,
)
from src.utils import load_json


def test_1_external_counterevidence_refutes_regardless_of_score():
    """Test 1: Confirmed external counterevidence produces REFUTED even with high scores."""
    candidate = {
        "evidence_cell_id": "cell_high_score_001",
        "subject": "Deep learning IoT IDS",
        "missing_capability": "Zero-day attacks",
        "impact_score": 0.95,
        "novelty_score": 0.99,
        "confidence_score": 0.98,
        "supporting_paper_count": 5,
    }
    external_record = {
        "verdict": "refuted_by_external_literature",
        "refuting_papers": [
            {"id": "p_counter_1", "title": "Zero-Day Attack Defense in IoT", "year": 2026}
        ],
        "confirmed_hit_count": 1,
    }
    author_record = {"corroboration_count": 3}
    saturation_summary = {"verdict": "SATURATED"}

    decision = decide_hypothesis(
        candidate=candidate,
        external_record=external_record,
        author_record=author_record,
        saturation_summary=saturation_summary,
    )

    assert decision.epistemic_disposition == EpistemicDisposition.REFUTED
    assert decision.action == AutonomousAction.REJECT
    assert decision.counterevidence_count == 1
    assert "counterevidence" in decision.decision_reason.lower()


def test_2_absence_corroborated_and_saturated_yields_evidence_supported():
    """Test 2: Zero counterevidence + independent author support + SATURATED corpus -> EVIDENCE_SUPPORTED."""
    candidate = {
        "evidence_cell_id": "cell_sat_002",
        "subject": "Lightweight IDS",
        "missing_capability": "On-chip memory bounds",
        "supporting_paper_count": 2,
    }
    external_record = {
        "verdict": "absence_corroborated",
        "refuting_papers": [],
        "confirmed_hit_count": 0,
    }
    author_record = {
        "corroboration_count": 2,
        "circular_count": 0,
    }
    saturation_summary = {
        "verdict": "SATURATED",
        "heaps_law": {"beta": 0.58, "r_squared": 0.99},
        "discovery_decay_rate": 0.65,
    }

    decision = decide_hypothesis(
        candidate=candidate,
        external_record=external_record,
        author_record=author_record,
        saturation_summary=saturation_summary,
    )

    assert decision.epistemic_disposition == EpistemicDisposition.EVIDENCE_SUPPORTED
    assert decision.action == AutonomousAction.RETAIN_AND_SYNTHESIZE
    assert decision.corpus_coverage_status == "SATURATED"
    assert decision.abstention_reason is None


def test_3_absence_corroborated_and_unsaturated_yields_evidence_supported_but_open():
    """Test 3: Zero counterevidence + independent author support + UNSATURATED corpus -> EVIDENCE_SUPPORTED_BUT_OPEN."""
    candidate = {
        "evidence_cell_id": "cell_3f81675bec9274a8",
        "subject": "machine learning-driven intrusion detection frameworks",
        "missing_capability": "computational burdens",
        "supporting_paper_count": 1,
    }
    external_record = {
        "verdict": "absence_corroborated",
        "refuting_papers": [],
        "confirmed_hit_count": 0,
    }
    author_record = {
        "corroboration_count": 3,
        "circular_count": 0,
    }
    saturation_summary = {
        "verdict": "UNSATURATED",
        "heaps_law": {"beta": 0.90, "r_squared": 0.998},
        "discovery_decay_rate": 0.158,
    }

    decision = decide_hypothesis(
        candidate=candidate,
        external_record=external_record,
        author_record=author_record,
        saturation_summary=saturation_summary,
    )

    assert decision.epistemic_disposition == EpistemicDisposition.EVIDENCE_SUPPORTED_BUT_OPEN
    assert decision.action == AutonomousAction.RETAIN_WITH_WARNING
    assert decision.corpus_coverage_status == "UNSATURATED"
    assert "UNSATURATED" in decision.decision_reason


def test_4_external_probe_failure_fails_closed_to_review_required():
    """Test 4: Network/API failure produces REVIEW_REQUIRED under fail-closed policy."""
    candidate = {
        "evidence_cell_id": "cell_timeout_004",
        "subject": "IoT security",
        "missing_capability": "latency",
    }
    external_record = {
        "verdict": "verification_failed",
        "reason": "HTTP 429 Too Many Requests; rate limited",
    }
    author_record = {"corroboration_count": 5}
    saturation_summary = {"verdict": "SATURATED"}

    decision = decide_hypothesis(
        candidate=candidate,
        external_record=external_record,
        author_record=author_record,
        saturation_summary=saturation_summary,
    )

    assert decision.epistemic_disposition == EpistemicDisposition.REVIEW_REQUIRED
    assert decision.action == AutonomousAction.ABSTAIN_AND_ESCALATE
    assert decision.abstention_reason is not None
    assert "failed" in decision.abstention_reason.lower()


def test_5_circular_evidence_does_not_increment_independent_corroboration():
    """Test 5: Source disjointness check prevents same-source statements from corroborating."""
    candidate_sources = ["paper_101", "paper_102"]
    circular_sources = ["paper_102", "paper_103"]
    independent_sources = ["paper_201", "paper_202"]

    assert not check_source_disjoint(circular_sources, candidate_sources)
    assert check_source_disjoint(independent_sources, candidate_sources)


def test_6_missing_kg_edge_alone_never_yields_supported_disposition():
    """Test 6: Pure graph absence without external verification must route to REVIEW_REQUIRED."""
    candidate = {
        "evidence_cell_id": "cell_kg_only_006",
        "type": "missing_link",
        "head": "ConceptA",
        "tail": "ConceptB",
        "_queue": "automatically_eligible",
    }

    # No external record provided
    decision = decide_hypothesis(candidate=candidate)

    assert decision.epistemic_disposition == EpistemicDisposition.REVIEW_REQUIRED
    assert decision.action == AutonomousAction.ABSTAIN_AND_ESCALATE
    assert decision.abstention_reason is not None


def test_7_extractor_miss_rate_bounds_uncertainty_without_inverse_fallacy():
    """Test 7: Extractor miss rate is stored as a diagnostic bound, not P(gap | missing_edge)."""
    candidate = {
        "evidence_cell_id": "cell_007",
        "subject": "model",
        "missing_capability": "efficiency",
    }
    extractor_summary = {
        "recall": 0.7778,
        "miss_rate": 0.2222,
        "precision": 0.875,
        "f1_score": 0.8235,
    }

    decision = decide_hypothesis(
        candidate=candidate,
        extractor_summary=extractor_summary,
    )

    rel = decision.extractor_reliability
    assert rel["recall"] == 0.7778
    assert rel["miss_rate"] == 0.2222
    assert "cannot independently establish" in rel["risk_interpretation"]


def test_8_decision_provenance_is_fully_preserved():
    """Test 8: The HypothesisDecision preserves external probes, refuting papers, and timestamps."""
    candidate = {
        "evidence_cell_id": "cell_prov_008",
        "subject": "IDS",
        "missing_capability": "speed",
        "supporting_paper_ids": ["p1", "p2"],
    }
    external_record = {
        "verdict": "refuted_by_external_literature",
        "probes": [
            {"source": "openalex", "query": '("IDS" AND "speed")', "ok": True, "raw_count": 0},
            {"source": "semantic_scholar", "query": '("IDS" AND "speed")', "ok": True, "raw_count": 40},
        ],
        "refuting_papers": [
            {"id": "p_ref_1", "title": "Speeding Up IDS", "year": 2025, "doi": "10.1234/test"}
        ],
    }

    decision = decide_hypothesis(candidate=candidate, external_record=external_record)
    d_dict = decision.to_dict()

    assert "provenance" in d_dict
    assert len(d_dict["provenance"]["external_probes"]) == 2
    assert d_dict["provenance"]["refuting_papers"][0]["doi"] == "10.1234/test"
    assert d_dict["provenance"]["supporting_paper_ids"] == ["p1", "p2"]
    assert d_dict["timestamp"] != ""


def test_9_frozen_experiment_reproduces_calibrated_triage():
    """Test 9: Historical IoT cybersecurity run reproduces exactly 4 REFUTED and 1 EVIDENCE_SUPPORTED_BUT_OPEN."""
    run_dir = Path("runs/deep_learning_iot_intrusion_de_20260831_114802/outputs")
    ext_path = run_dir / "external_verification.json"
    auth_path = run_dir / "author_stated_gaps.json"
    sat_path = run_dir / "corpus_saturation_report.json"
    recall_path = run_dir / "extractor_recall_report.json"

    assert ext_path.exists()
    assert auth_path.exists()
    assert sat_path.exists()
    assert recall_path.exists()

    ext_data = load_json(ext_path)
    auth_data = load_json(auth_path)
    sat_data = load_json(sat_path)
    recall_data = load_json(recall_path)

    ext_candidates = ext_data.get("candidates", [])
    auth_candidates = auth_data.get("candidates", [])
    sat_summary = sat_data.get("summary", {})
    recall_summary = recall_data.get("summary", {})

    assert len(ext_candidates) == 5
    assert sat_summary.get("verdict") == "UNSATURATED"
    assert recall_summary.get("recall") == 0.7778
    assert recall_summary.get("miss_rate") == 0.2222

    auth_by_id = {c.get("evidence_cell_id"): c for c in auth_candidates}

    decisions: list[HypothesisDecision] = []
    for ext_cand in ext_candidates:
        cid = ext_cand.get("evidence_cell_id")
        auth_cand = auth_by_id.get(cid)
        decision = decide_hypothesis(
            candidate=ext_cand,
            external_record=ext_cand,
            author_record=auth_cand,
            saturation_summary=sat_summary,
            extractor_summary=recall_summary,
        )
        decisions.append(decision)

    refuted = [d for d in decisions if d.epistemic_disposition == EpistemicDisposition.REFUTED]
    open_supported = [
        d for d in decisions
        if d.epistemic_disposition == EpistemicDisposition.EVIDENCE_SUPPORTED_BUT_OPEN
    ]
    strictly_supported = [
        d for d in decisions
        if d.epistemic_disposition == EpistemicDisposition.EVIDENCE_SUPPORTED
    ]
    review_req = [
        d for d in decisions
        if d.epistemic_disposition == EpistemicDisposition.REVIEW_REQUIRED
    ]

    # Exactly 4 candidates refuted by external literature
    assert len(refuted) == 4
    refuted_caps = {d.missing_capability.lower() for d in refuted}
    assert any("zero-day" in cap for cap in refuted_caps)
    assert any("protocol" in cap for cap in refuted_caps)
    assert any("industrial" in cap for cap in refuted_caps)
    assert any("adversarial" in cap for cap in refuted_caps)

    # Exactly 1 surviving candidate: computational burdens / edge constraints
    assert len(open_supported) == 1
    surviving = open_supported[0]
    assert "computational" in surviving.missing_capability.lower()
    assert surviving.independent_corroboration_count == 3
    assert surviving.counterevidence_count == 0
    assert surviving.action == AutonomousAction.RETAIN_WITH_WARNING

    # Because corpus is UNSATURATED, strictly supported count is 0
    assert len(strictly_supported) == 0
    assert len(review_req) == 0
