"""Create a blinded, stratified expert-review packet from a validation audit.

The CSV deliberately omits the system verdict and gate reasons.  The JSON
manifest preserves that mapping for later agreement and precision analysis.
No human labels are fabricated by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _identity(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": candidate.get("type"),
        "subject": candidate.get("subject"),
        "missing_capability": candidate.get("missing_capability"),
        "head": candidate.get("head"),
        "tail": candidate.get("tail"),
        "concept": candidate.get("concept"),
        "community_id": candidate.get("community_id"),
    }


def _blind_id(candidate: dict[str, Any], seed: int) -> str:
    payload = json.dumps(_identity(candidate), sort_keys=True, ensure_ascii=False)
    return "C-" + hashlib.sha256(f"{seed}|{payload}".encode()).hexdigest()[:10].upper()


def _statement(candidate: dict[str, Any]) -> str:
    if candidate.get("type") == "missing_link":
        return (
            "Assess whether evidence is inadequate for the relationship between "
            f"'{candidate.get('head', '')}' and '{candidate.get('tail', '')}'."
        )
    if candidate.get("type") == "evidence_gap":
        return (
            "Assess whether the reported limitation remains an open research gap: "
            f"'{candidate.get('subject', '')}' - '{candidate.get('missing_capability', '')}'."
        )
    return str(candidate.get("description") or candidate.get("concept") or "")


def _stratified_rejected(candidates: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        validation = candidate.get("validation", {})
        reason = next(iter(validation.get("reasons", [])), "no_recorded_reason")
        groups[(str(candidate.get("type")), str(reason))].append(candidate)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, help="gap_validation_audit.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rejected-sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    candidates = list(audit.get("candidates", []))
    non_rejected = [
        item for item in candidates
        if item.get("validation", {}).get("status") != "rejected"
    ]
    rejected = [
        item for item in candidates
        if item.get("validation", {}).get("status") == "rejected"
    ]
    selected = non_rejected + _stratified_rejected(rejected, args.rejected_sample, args.seed)
    random.Random(args.seed).shuffle(selected)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "blinded_expert_review_packet.csv"
    manifest_path = output_dir / "blinded_expert_review_manifest.json"
    fields = [
        "candidate_id", "signal_type", "candidate_statement", "source_evidence",
        "supporting_paper_ids", "closure_titles", "is_research_gap_yes_no",
        "already_addressed_yes_no", "evidence_sufficient_1_5",
        "importance_1_5", "actionability_1_5", "reviewer_id", "notes",
    ]
    manifest = []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate in selected:
            validation = candidate.get("validation", {})
            evidence = " | ".join(
                str(item.get("evidence", ""))
                for item in candidate.get("source_evidence", [])[:3]
            )
            closure_titles = " | ".join(
                str(hit.get("title", ""))
                for hit in validation.get("closure_hits", [])[:5]
            )
            candidate_id = _blind_id(candidate, args.seed)
            writer.writerow({
                "candidate_id": candidate_id,
                "signal_type": candidate.get("type", ""),
                "candidate_statement": _statement(candidate),
                "source_evidence": evidence,
                "supporting_paper_ids": ";".join(validation.get("supporting_paper_ids", [])),
                "closure_titles": closure_titles,
            })
            manifest.append({
                "candidate_id": candidate_id,
                "candidate_identity": _identity(candidate),
                "system_status": validation.get("status"),
                "system_reasons": validation.get("reasons", []),
                "selection": (
                    "all_non_rejected"
                    if validation.get("status") != "rejected"
                    else "stratified_rejected_sample"
                ),
            })
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "seed": args.seed,
        "review_items": len(manifest),
        "instructions": (
            "Use at least three independent domain experts. Keep the manifest "
            "hidden until reviews are frozen, then calculate agreement and the "
            "confusion matrix against system_status."
        ),
        "manifest": manifest,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "packet": str(csv_path),
        "manifest": str(manifest_path),
        "review_items": len(manifest),
        "non_rejected": len(non_rejected),
        "rejected_sample": len(manifest) - len(non_rejected),
    }, indent=2))


if __name__ == "__main__":
    main()
