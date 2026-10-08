"""Scope-aware control over reviewed evidence; never certifies global novelty.

This module consumes annotations, not raw-text entailment predictions. A quote
check proves provenance only. Human semantic review remains a trusted input.
The historical ESV-Gap decision engine is intentionally not replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any


def _quoted(item: dict, source: str) -> bool:
    span = item.get("span")
    return (item.get("reviewed") is True and isinstance(span, str)
            and bool(span.strip()) and span in source)


def _number(value: Any) -> bool:
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and isfinite(value))


def _validate_contract(contract: dict) -> list[str]:
    errors = []
    if not isinstance(contract.get("task"), str) or not contract["task"].strip():
        errors.append("missing_task")
    if not isinstance(contract.get("domain"), str) or not contract["domain"].strip():
        errors.append("missing_domain")
    try:
        date.fromisoformat(contract.get("cutoff", ""))
    except (TypeError, ValueError):
        errors.append("invalid_cutoff")
    if contract.get("scope_reviewed") is not True:
        errors.append("unreviewed_scope")
    conditions = contract.get("conditions")
    if not isinstance(conditions, list) or not conditions or any(
            not isinstance(c, str) or not c.strip() for c in conditions):
        errors.append("missing_conditions")
    elif len(set(conditions)) != len(conditions):
        errors.append("duplicate_conditions")
    requirements = contract.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        errors.append("missing_requirements")
    else:
        names = []
        for req in requirements:
            if not isinstance(req, dict):
                errors.append("invalid_requirement")
                continue
            if isinstance(req.get("name"), str):
                names.append(req["name"])
            if (not isinstance(req.get("name"), str) or not req["name"].strip()
                    or req.get("op") not in ("<=", ">=")
                    or not _number(req.get("bound"))
                    or not isinstance(req.get("unit"), str) or not req["unit"].strip()):
                errors.append("invalid_requirement")
        if len(set(names)) != len(names):
            errors.append("duplicate_requirements")
    return sorted(set(errors))


def _search_complete(ledger: dict, cutoff: str) -> tuple[bool, list[str]]:
    """Require every predeclared probe, not just an empty returned list."""
    expected = ledger.get("required_probe_ids")
    if not isinstance(expected, list) or not expected or any(
            not isinstance(x, str) or not x.strip() for x in expected):
        return False, ["missing_probe_plan"]
    if len(set(expected)) != len(expected):
        return False, ["duplicate_probe_plan"]
    rows = ledger.get("probes", [])
    if not isinstance(rows, list):
        return False, ["invalid_probe_records"]
    errors = []
    by_id: dict[str, list] = {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("invalid_probe_record")
            continue
        by_id.setdefault(row.get("probe_id", ""), []).append(row)
    if ledger.get("cutoff") != cutoff:
        errors.append("search_cutoff_mismatch")
    for pid in expected:
        found = by_id.get(pid, [])
        if len(found) != 1:
            errors.append(f"missing_or_duplicate_probe:{pid}")
            continue
        row = found[0]
        if (row.get("status") != "completed"
                or row.get("pagination_complete") is not True
                or row.get("semantic_review_complete") is not True
                or not str(row.get("query") or "").strip()
                or not str(row.get("provider") or "").strip()
                or not str(row.get("snapshot_id") or "").strip()):
            errors.append(f"incomplete_probe:{pid}")
    return not errors, sorted(set(errors))


def evaluate_scope(contract: dict, evidence: list[dict], ledger: dict) -> dict:
    """Produce residual obligations, a bounded disposition and an audit hash.

    Evidence is grouped by one (paper_id, experiment_id), never merged across
    records. All scope facets and measurements must carry reviewed source spans.
    A completed search with no witness still does not establish a research gap.
    """
    errors = _validate_contract(contract)
    complete, search_errors = _search_complete(ledger, contract.get("cutoff", ""))
    result = {"schema_version": 1, "novelty_established": False,
              "claim_boundary": "relative_to_supplied_reviewed_annotations",
              "search_protocol_complete": complete, "contract_errors": errors,
              "search_errors": search_errors, "traces": []}
    if errors:
        result.update(disposition="REVIEW_REQUIRED", action="ABSTAIN_AND_ESCALATE")
    else:
        key_counts = Counter(
            (row["paper_id"], row["experiment_id"]) for row in evidence
            if isinstance(row, dict) and isinstance(row.get("paper_id"), str)
            and isinstance(row.get("experiment_id"), str))
        for record in evidence:
            if not isinstance(record, dict):
                result["traces"].append({"status": "unchecked_evidence"})
                continue
            pid = record.get("paper_id")
            eid = record.get("experiment_id")
            trace = {"paper_id": pid, "experiment_id": eid, "residuals": []}
            result["traces"].append(trace)
            try:
                available = date.fromisoformat(record.get("available_on", ""))
            except (TypeError, ValueError):
                trace["status"] = "unchecked_evidence"
                trace["reason"] = "missing_or_invalid_availability"
                continue
            if available > date.fromisoformat(contract["cutoff"]):
                trace["status"] = "excluded_future"
                continue
            key = (pid, eid)
            if (not isinstance(pid, str) or not pid.strip()
                    or not isinstance(eid, str) or not eid.strip()
                    or key_counts[key] != 1 or record.get("metadata_reviewed") is not True
                    or not isinstance(record.get("source_text"), str)
                    or not record["source_text"].strip()):
                trace["status"] = "unchecked_evidence"
                trace["reason"] = "invalid_or_duplicate_provenance"
                continue
            source = record["source_text"]
            if record.get("experiment_reviewed") is not True:
                trace["status"] = "unchecked_evidence"
                trace["reason"] = "experiment_membership_unreviewed"
                continue
            facets = record.get("facets", {})
            if not isinstance(facets, dict):
                facets = {}
            for facet in ["task", "domain"]:
                annotation = facets.get(facet, {})
                if not isinstance(annotation, dict):
                    annotation = {}
                state = ("satisfied" if annotation.get("value") == contract[facet]
                         else "different_scope")
                if not _quoted(annotation, source):
                    state = "unreviewed_or_unquoted"
                trace["residuals"].append({"facet": facet, "state": state})
            conditions = record.get("conditions", {})
            if not isinstance(conditions, dict):
                conditions = {}
            for condition in contract["conditions"]:
                annotation = conditions.get(condition, {})
                state = "satisfied"
                if not isinstance(annotation, dict) or not _quoted(annotation, source):
                    state = "unreviewed_or_unquoted"
                elif annotation.get("value") is not True:
                    state = "not_satisfied"
                trace["residuals"].append({"facet": condition, "state": state})
            measurements = record.get("measurements", {})
            if not isinstance(measurements, dict):
                measurements = {}
            for req in contract["requirements"]:
                m = measurements.get(req["name"])
                if not isinstance(m, dict):
                    state = "unreported"
                elif not _quoted(m, source):
                    state = "unreviewed_or_unquoted"
                elif m.get("unit") != req["unit"] or not _number(m.get("value")):
                    state = "incomparable"
                else:
                    ok = (m["value"] <= req["bound"] if req["op"] == "<="
                          else m["value"] >= req["bound"])
                    state = "satisfied" if ok else "not_satisfied"
                trace["residuals"].append({"facet": req["name"], "state": state})
            states = {r["state"] for r in trace["residuals"]}
            if states == {"satisfied"}:
                trace["status"] = "closes_scope"
            elif "unreviewed_or_unquoted" in states:
                trace["status"] = "unchecked_evidence"
            elif "different_scope" in states:
                trace["status"] = "different_scope"
            else:
                trace["status"] = "partial_evidence"
        statuses = {t["status"] for t in result["traces"]}
        if "closes_scope" in statuses:
            disposition, action = "CLOSED_WITHIN_SCOPE", "SUPPRESS_CANDIDATE"
        elif not complete or "unchecked_evidence" in statuses:
            disposition, action = "REVIEW_REQUIRED", "ABSTAIN_AND_ESCALATE"
        elif "partial_evidence" in statuses:
            disposition, action = "PARTIAL_EVIDENCE", "REQUEST_RESIDUAL_EVIDENCE"
        else:
            disposition, action = "OPEN_FOR_REVIEW", "ABSTAIN_AND_ESCALATE"
        result.update(disposition=disposition, action=action)
    def hashable(value):
        if isinstance(value, float) and not isfinite(value):
            return {"invalid_nonfinite_number": repr(value)}
        if isinstance(value, dict):
            return {k: hashable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [hashable(v) for v in value]
        return value
    canonical = json.dumps(hashable({"contract": contract, "evidence": evidence,
                                    "ledger": ledger}),
                           ensure_ascii=False, sort_keys=True, allow_nan=False)
    result["input_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = evaluate_scope(payload["contract"], payload.get("evidence", []),
                            payload.get("ledger", {}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"disposition": result["disposition"],
                      "novelty_established": False}))


if __name__ == "__main__":
    main()
