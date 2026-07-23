"""Create blinded candidate-review packets from real system outputs.

The module never generates candidates or scores. It randomly assigns opaque
review IDs to supplied candidate files and writes the system mapping separately
so reviewers cannot see a method label during rating.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


RATING_COLUMNS = (
    "source_support_1_to_5",
    "claim_clarity_1_to_5",
    "novelty_after_closure_1_to_5",
    "importance_1_to_5",
    "actionability_1_to_5",
    "feasibility_1_to_5",
    "already_addressed_yes_no_uncertain",
    "unsupported_or_hallucinated_evidence_yes_no_uncertain",
    "reviewer_id",
    "comments",
)


def _read_candidates(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Candidate input must be a JSON array: {path}")
    return [item for item in data if isinstance(item, dict) and str(item.get("Claim", "")).strip()]


def build_blind_packet(inputs: dict[str, Path], *, seed: int) -> tuple[list[dict], list[dict]]:
    """Return a shuffled reviewer packet and the private unblinding key."""
    source_rows = []
    for system_id, path in sorted(inputs.items()):
        for source_index, candidate in enumerate(_read_candidates(path)):
            source_rows.append((system_id, path, source_index, candidate))
    randomizer = random.Random(seed)
    randomizer.shuffle(source_rows)
    packet, key = [], []
    for blind_index, (system_id, path, source_index, candidate) in enumerate(source_rows, start=1):
        blind_id = f"blind-{blind_index:04d}"
        packet.append({
            "blind_candidate_id": blind_id,
            "claim": candidate.get("Claim", ""),
            "grounds": candidate.get("Grounds", ""),
            "warrant": candidate.get("Warrant", ""),
            "bucket": candidate.get("Bucket", ""),
            **{column: "" for column in RATING_COLUMNS},
        })
        key.append({
            "blind_candidate_id": blind_id,
            "system_id": system_id,
            "source_path": str(path),
            "source_index": source_index,
        })
    return packet, key


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare blinded expert-review candidate packet.")
    parser.add_argument(
        "--input", action="append", required=True, metavar="SYSTEM=PATH",
        help="Repeat once per system output JSON, e.g. kgtabi=data/.../kgtabi_gaps.json",
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    inputs: dict[str, Path] = {}
    for item in args.input:
        if "=" not in item:
            parser.error("--input must have SYSTEM=PATH form")
        system_id, raw_path = item.split("=", 1)
        if not system_id or not raw_path or system_id in inputs:
            parser.error("System IDs must be unique and paths non-empty")
        inputs[system_id] = Path(raw_path)
    packet, key = build_blind_packet(inputs, seed=args.seed)
    _write_csv(args.packet, packet, list(packet[0]) if packet else ["blind_candidate_id"])
    _write_csv(args.key, key, ["blind_candidate_id", "system_id", "source_path", "source_index"])
    print(f"[+] Wrote {len(packet)} blinded candidates to {args.packet}")
    print(f"[+] Wrote separate unblinding key to {args.key}")


if __name__ == "__main__":
    main()
