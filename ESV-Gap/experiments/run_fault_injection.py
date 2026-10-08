"""Deterministic fault-injection validation for the autonomous decision engine.

Validates that unrecoverable provider failures, rate-limits, circular provenance,
or incomplete evidence strictly fail closed to REVIEW_REQUIRED and trigger
autonomous abstention (ABSTAIN_AND_ESCALATE).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.autonomous_reasoning import (
    AutonomousAction,
    EpistemicDisposition,
    decide_hypothesis,
)


def run_fault_injection_suite() -> dict[str, Any]:
    base_candidate = {
        "evidence_cell_id": "cell_fault_test_001",
        "subject": "Deep learning IoT IDS",
        "missing_capability": "Resource constraint edge deployment",
        "_queue": "review_required",
        "supporting_paper_count": 2,
        "supporting_paper_ids": ["paper_orig_1", "paper_orig_2"],
    }

    scenarios = [
        {
            "id": "SCENARIO_1_TIMEOUT",
            "name": "External Provider Timeout",
            "description": "Semantic Scholar / OpenAlex probe times out after max retries.",
            "external_record": {
                "verdict": "verification_failed",
                "error": "ExternalProbeError: https://api.semanticscholar.org timed out after 15.0s",
                "probes": [{"ok": False, "error": "timeout"}],
            },
            "author_record": {"corroboration_count": 2, "circular_count": 0},
            "saturation": {"verdict": "SATURATED"},
        },
        {
            "id": "SCENARIO_2_HTTP_429",
            "name": "HTTP 429 Quota Exhaustion",
            "description": "Scholarly index throttles requests with HTTP 429.",
            "external_record": {
                "verdict": "verification_failed",
                "error": "ExternalProbeError: index throttled with HTTP 429",
                "probes": [{"ok": False, "error": "HTTP 429"}],
            },
            "author_record": {"corroboration_count": 2, "circular_count": 0},
            "saturation": {"verdict": "SATURATED"},
        },
        {
            "id": "SCENARIO_3_INCOMPLETE_PAGINATION",
            "name": "Incomplete / Truncated Pagination",
            "description": "Index pagination breaks midway returning partial data.",
            "external_record": {
                "verdict": "verification_failed",
                "error": "ExternalProbeError: pagination interrupted at page 2",
                "probes": [{"ok": False, "error": "incomplete pagination"}],
            },
            "author_record": {"corroboration_count": 2, "circular_count": 0},
            "saturation": {"verdict": "SATURATED"},
        },
        {
            "id": "SCENARIO_4_MALFORMED_RESPONSE",
            "name": "Malformed JSON Payload",
            "description": "Upstream scholarly API returns corrupted or non-decodable JSON.",
            "external_record": {
                "verdict": "verification_failed",
                "error": "JSONDecodeError: Unterminated string starting at line 1",
                "probes": [{"ok": False, "error": "json decode error"}],
            },
            "author_record": {"corroboration_count": 2, "circular_count": 0},
            "saturation": {"verdict": "SATURATED"},
        },
        {
            "id": "SCENARIO_5_CIRCULAR_CORROBORATION_ONLY",
            "name": "Circular Provenance Only",
            "description": "External search passes, but all local corroborating papers originate from the candidate's own source paper.",
            "external_record": {
                "verdict": "absence_corroborated",
                "probes": [{"ok": True, "raw_count": 0, "confirmed_hit_count": 0}],
            },
            "author_record": {
                "corroboration_count": 0,
                "circular_count": 2,
                "circular": [{"source_papers": ["paper_orig_1"]}],
            },
            "saturation": {"verdict": "SATURATED"},
        },
        {
            "id": "SCENARIO_6_MISSING_PROVENANCE",
            "name": "Missing Evidence Provenance",
            "description": "Candidate lacks supporting paper IDs or evidence extraction anchors.",
            "candidate_override": {
                "supporting_paper_count": 0,
                "supporting_paper_ids": [],
            },
            "external_record": {
                "verdict": "absence_corroborated",
                "probes": [{"ok": True, "raw_count": 0, "confirmed_hit_count": 0}],
            },
            "author_record": {"corroboration_count": 2, "circular_count": 0},
            "saturation": {"verdict": "SATURATED"},
        },
    ]

    results = []
    for sc in scenarios:
        cand = dict(base_candidate)
        if "candidate_override" in sc:
            cand.update(sc["candidate_override"])

        decision = decide_hypothesis(
            candidate=cand,
            external_record=sc["external_record"],
            author_record=sc["author_record"],
            saturation_summary=sc["saturation"],
        )

        passed = (
            decision.epistemic_disposition == EpistemicDisposition.REVIEW_REQUIRED
            and decision.action == AutonomousAction.ABSTAIN_AND_ESCALATE
        )

        results.append({
            "scenario_id": sc["id"],
            "name": sc["name"],
            "description": sc["description"],
            "epistemic_disposition": decision.epistemic_disposition.value,
            "action": decision.action.value,
            "abstention_reason": decision.abstention_reason,
            "fail_closed_passed": passed,
        })

    summary = {
        "total_scenarios": len(scenarios),
        "passed_scenarios": sum(1 for r in results if r["fail_closed_passed"]),
        "fail_closed_guarantee": all(r["fail_closed_passed"] for r in results),
        "results": results,
    }

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fault_injection_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = run_fault_injection_suite()
    print(f"Fault Injection Validation: {res['passed_scenarios']}/{res['total_scenarios']} passed.")
    print(f"Fail-Closed Guarantee Verified: {res['fail_closed_guarantee']}")
