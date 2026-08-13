"""Create a deterministic, blinded expert-review packet without fake labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def candidate_text(item):
    if item["type"] == "missing_link":
        return f"Assess possible under-studied relation: {item.get('head')} -> {item.get('tail')}"
    if item["type"] == "temporal_decay":
        return f"Assess whether the normalised decline of '{item.get('concept')}' motivates new research"
    concepts = ", ".join(map(str, item.get("key_concepts", [])[:5]))
    return f"Assess whether this weakly connected topic cluster motivates new research: {concepts}"


def blinded_id(item, salt):
    identity = json.dumps(
        {
            "type": item.get("type"),
            "head": item.get("head"),
            "tail": item.get("tail"),
            "concept": item.get("concept"),
            "community_id": item.get("community_id"),
        },
        sort_keys=True,
    )
    return "C-" + hashlib.sha256((salt + identity).encode()).hexdigest()[:10].upper()


def stratified_rejected_sample(candidates, sample_size, seed):
    groups = defaultdict(list)
    for item in candidates:
        validation = item["validation"]
        primary_reason = next((reason for reason in validation["reasons"] if reason != "possible_prior_coverage_found_in_local_corpus"), validation["reasons"][0] if validation["reasons"] else "none")
        groups[(item["type"], primary_reason)].append(item)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < sample_size and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < sample_size:
                selected.append(groups[key].pop())
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="outputs/microservices_corpus_validation_v3.json")
    parser.add_argument("--review-csv", default="outputs/expert_review_packet.csv")
    parser.add_argument("--manifest", default="outputs/expert_review_manifest.json")
    parser.add_argument("--rejected-sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    candidates = report["candidates"]
    non_rejected = [item for item in candidates if item["validation"]["status"] != "rejected"]
    rejected = [item for item in candidates if item["validation"]["status"] == "rejected"]
    sample = non_rejected + stratified_rejected_sample(rejected, args.rejected_sample, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(sample)

    review_path = Path(args.review_csv)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    fields = [
        "candidate_id", "signal_type", "candidate_statement", "evidence_paper_ids",
        "retrieval_hit_ids", "source_support_1_5", "novelty_after_retrieval_1_5",
        "importance_1_5", "actionability_1_5", "feasibility_1_5",
        "already_addressed_yes_no", "extraction_error_yes_no", "review_minutes", "notes",
    ]
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in sample:
            validation = item["validation"]
            candidate_id = blinded_id(item, str(args.seed))
            writer.writerow({
                "candidate_id": candidate_id,
                "signal_type": item["type"],
                "candidate_statement": candidate_text(item),
                "evidence_paper_ids": ";".join(validation.get("supporting_paper_ids", [])),
                "retrieval_hit_ids": ";".join(str(hit.get("paper_id")) for hit in validation.get("closure_hits", [])),
            })
            manifest.append({
                "candidate_id": candidate_id,
                "system_status": validation["status"],
                "selection": "all_non_rejected" if validation["status"] != "rejected" else "stratified_rejected_sample",
                "reasons": validation["reasons"],
            })
    Path(args.manifest).write_text(
        json.dumps({"seed": args.seed, "review_items": len(sample), "manifest": manifest}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"review_items": len(sample), "non_rejected": len(non_rejected), "rejected_sample": len(sample) - len(non_rejected)}, indent=2))


if __name__ == "__main__":
    main()
