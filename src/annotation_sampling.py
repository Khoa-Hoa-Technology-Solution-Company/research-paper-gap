"""Create auditable, blinded review packets without fabricating judgments."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

from src import config


REVIEW_FIELDS = [
    "source_support", "novel_after_closure", "importance", "actionability",
    "feasibility", "already_addressed", "reviewer_id", "comments",
]


def sample_records(records: list[dict], size: int, seed: int = 42) -> list[dict]:
    randomizer = random.Random(seed)
    return randomizer.sample(records, min(size, len(records)))


def write_triple_audit(triples: list[dict], output: str, size: int, seed: int) -> None:
    rows = []
    for index, triple in enumerate(sample_records(triples, size, seed)):
        rows.append({
            "item_id": f"triple-{index:03d}", "subject": triple.get("subject"),
            "relation": triple.get("relation"), "object": triple.get("object"),
            "year": triple.get("year"), "confidence": triple.get("confidence"),
            "is_factually_supported": "", "is_correctly_typed": "", "comments": "",
        })
    _write_csv(output, rows)


def write_candidate_packet(gaps: list[dict], output: str, size: int, seed: int) -> None:
    rows = []
    for index, gap in enumerate(sample_records(gaps, size, seed)):
        row = {
            "item_id": f"candidate-{index:03d}", "claim": gap.get("Claim"),
            "grounds": gap.get("Grounds"), "warrant": gap.get("Warrant"),
            "source_signal": gap.get("source"),
        }
        row.update({field: "" for field in REVIEW_FIELDS})
        rows.append(row)
    _write_csv(output, rows)


def _write_csv(output: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["item_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create blinded KG-TABI annotation packets.")
    parser.add_argument("--triples", default=os.path.join(config.TRIPLES_DIR, "resolved_triples.json"))
    parser.add_argument("--gaps", default=os.path.join(config.GAPS_DIR, "kgtabi_gaps.json"))
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if os.path.exists(args.triples):
        with open(args.triples, encoding="utf-8") as handle:
            write_triple_audit(json.load(handle), os.path.join(config.DATA_DIR, "triple_audit_packet.csv"), args.size, args.seed)
    if os.path.exists(args.gaps):
        with open(args.gaps, encoding="utf-8") as handle:
            write_candidate_packet(json.load(handle), os.path.join(config.DATA_DIR, "candidate_review_packet.csv"), args.size, args.seed)


if __name__ == "__main__":
    main()
