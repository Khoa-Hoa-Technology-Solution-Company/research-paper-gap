"""Prepare an auditable review ledger for every post-gate review candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def candidate_identity(candidate: dict) -> dict:
    gap_type = candidate.get("type")
    if gap_type == "missing_link":
        return {
            "type": gap_type,
            "head": candidate.get("head"),
            "tail": candidate.get("tail"),
        }
    if gap_type == "orphan_cluster":
        return {
            "type": gap_type,
            "members": sorted(map(str, candidate.get("members", []))),
        }
    return {"type": gap_type, "concept": candidate.get("concept")}


def review_key(candidate: dict) -> str:
    payload = json.dumps(candidate_identity(candidate), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"post_gate_{digest}"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(outputs: Path) -> dict:
    queue = _load(outputs / "review_required_gaps.json")
    ranked = _load(outputs / "gaps_ranked_top.json")
    prior = _load(outputs / "expert_reviews.json")

    ranked_by_identity = {
        json.dumps(candidate_identity(item), sort_keys=True, ensure_ascii=False): item
        for item in ranked
    }
    records = []
    reviews = {}
    position = 0
    for category in ("missing_links", "orphan_clusters", "temporal_decay"):
        for candidate in queue.get(category, []):
            position += 1
            identity = candidate_identity(candidate)
            identity_json = json.dumps(identity, sort_keys=True, ensure_ascii=False)
            ranked_match = ranked_by_identity.get(identity_json)
            prior_key = f"review_{ranked_match['rank']}" if ranked_match else None
            carried = prior.get("reviews", {}).get(prior_key) if prior_key else None
            decision = carried if carried in {"Accept", "Reject", "Modify"} else "Pending"
            key = review_key(candidate)
            reviews[key] = decision
            validation = candidate.get("validation", {})
            records.append({
                "queue_position": position,
                "review_key": key,
                "candidate_identity": identity,
                "candidate": candidate,
                "decision": decision,
                "decision_source": (
                    f"carried_from_expert_reviews:{prior_key}"
                    if decision != "Pending"
                    else "pending_author_review"
                ),
                "prior_rank": ranked_match.get("rank") if ranked_match else None,
                "ranking_score": validation.get("ranking_score"),
                "validation_reasons": validation.get("reasons", []),
                "supporting_paper_count": validation.get("supporting_paper_count", 0),
                "closure_hit_count": validation.get("closure_hit_count", 0),
                "reviewer_rationale": "",
            })

    counts = Counter(reviews.values())
    reviewed = sum(counts[key] for key in ("Accept", "Reject", "Modify"))
    return {
        "schema_version": 1,
        "queue_name": "post_gate_review_required",
        "reviewer": prior.get("reviewer", "author_internal"),
        "source_queue": "review_required_gaps.json",
        "prior_review_source": "expert_reviews.json",
        "candidate_count": len(records),
        "reviews": reviews,
        "summary": {key: counts[key] for key in ("Accept", "Reject", "Modify", "Pending")},
        "total_reviewed": reviewed,
        "complete": reviewed == len(records),
        "candidate_index": records,
        "notes": (
            "Only exact candidate-identity matches are carried from the prior top-30 review. "
            "Pending items require a human author decision; no model-generated label is inserted."
        ),
    }


def write_csv(report: dict, path: Path) -> None:
    rows = []
    for record in report["candidate_index"]:
        candidate = record["candidate"]
        identity = record["candidate_identity"]
        validation = candidate.get("validation", {})
        closure_titles = [
            str(hit.get("title", "")) for hit in validation.get("closure_hits", [])
        ]
        rows.append({
            "queue_position": record["queue_position"],
            "review_key": record["review_key"],
            "type": identity.get("type"),
            "candidate": json.dumps(identity, ensure_ascii=False, sort_keys=True),
            "description": candidate.get("description", ""),
            "ranking_score": record["ranking_score"],
            "supporting_paper_count": record["supporting_paper_count"],
            "closure_hit_count": record["closure_hit_count"],
            "closure_titles": " | ".join(closure_titles),
            "validation_reasons": " | ".join(record["validation_reasons"]),
            "decision": record["decision"],
            "decision_source": record["decision_source"],
            "reviewer_rationale": record["reviewer_rationale"],
        })
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True)
    args = parser.parse_args()
    outputs = Path(args.outputs)
    report = prepare(outputs)
    json_path = outputs / "post_gate_expert_reviews.json"
    csv_path = outputs / "post_gate_review_packet.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(report, csv_path)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "summary": report["summary"],
        "complete": report["complete"],
    }, indent=2))


if __name__ == "__main__":
    main()
