"""Score independent label agreement; do not invent ground truth or efficacy."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

LABELS = {"SAME_SCOPE_CLOSURE", "PARTIAL_EVIDENCE", "SUPPORTING_LIMITATION",
          "TOPICAL_ONLY", "INSUFFICIENT_EVIDENCE", "UNDERSPECIFIED_CANDIDATE"}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        iid = row["item_id"]
        if iid in result:
            raise ValueError(f"Duplicate item: {iid}")
        if row.get("evidence_label") not in LABELS:
            raise ValueError(f"Missing/invalid human label: {iid}")
        if not row.get("reviewer_id", "").strip():
            raise ValueError(f"Missing reviewer identity: {iid}")
        if row["evidence_label"] in {"SAME_SCOPE_CLOSURE", "PARTIAL_EVIDENCE", "SUPPORTING_LIMITATION"}:
            span = row.get("source_span", "")
            if not span.strip() or span not in row["evidence_abstract"]:
                raise ValueError(f"Unquoted abstract evidence: {iid}")
        if row["evidence_label"] == "SAME_SCOPE_CLOSURE" and (
                row.get("scope_sufficient") != "yes" or not row.get("experiment_id", "").strip()):
            raise ValueError(f"Closure requires scope and experiment identity: {iid}")
        result[iid] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).parent / "review_packet/organizer_manifest.json",
                        help="Frozen full packet or pilot manifest; inputs must cover its exact item set")
    args = parser.parse_args()
    a, b = read(args.reviewer_a), read(args.reviewer_b)
    expected_path = args.manifest
    expected = {r["item_id"] for r in json.loads(expected_path.read_text(encoding="utf-8"))}
    if not expected or set(a) != expected or set(b) != expected:
        raise ValueError("Review inputs must cover the full prepared item set")
    people_a = {r["reviewer_id"] for r in a.values()}
    people_b = {r["reviewer_id"] for r in b.values()}
    if people_a & people_b:
        raise ValueError("Reviewers must be independent identities")
    n = len(expected)
    ca = Counter(r["evidence_label"] for r in a.values())
    cb = Counter(r["evidence_label"] for r in b.values())
    agreements = sum(a[i]["evidence_label"] == b[i]["evidence_label"] for i in expected)
    po = agreements / n
    pe = sum(ca[label] * cb[label] for label in LABELS) / n**2
    result = {"status": "human_annotation_agreement_not_system_accuracy",
              "pairs": n, "observed_agreement": po,
              "cohens_kappa": (po - pe) / (1 - pe) if pe < 1 else None,
              "reviewer_a_counts": dict(ca), "reviewer_b_counts": dict(cb),
              "needs_adjudication": sorted(i for i in expected if a[i]["evidence_label"] != b[i]["evidence_label"]),
              "independence_warning": "Pairs share candidates/papers. No iid confidence interval computed.",
              "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [args.reviewer_a, args.reviewer_b, expected_path]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": n, "cohens_kappa": result["cohens_kappa"]}))


if __name__ == "__main__":
    main()
