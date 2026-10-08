"""Build local, blinded annotation inputs with raw-abstract BM25 retrieval.

No network/LLM use. No outcome labels are generated. Candidate generation still
uses archived extracted triples; only this evidence retriever is extraction-free.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).resolve().parent
ARCHIVE = PROJECT / "paper_icai2026/expanded_run"
RUNS = {
    "iot_intrusion_detection": "deep_learning_iot_intrusion_de_20260831_114802",
    "microservice_security": "security_of_microservices_20260806_114632",
    "mongodb_security": "security_of_mongodb_20260806_135447",
    "handwritten_math_recognition": "handwritten_mathematical_expre_20260823_215228",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifier(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def terms(text):
    return re.findall(r"[a-z0-9]+", text.casefold().replace("-", " "))


def retrieve(documents, query, cutoff, excluded, k=5):
    pool = [d for d in documents if int(d.get("year") or 0) <= cutoff
            and str(d.get("paperId") or d.get("paper_id") or "") not in excluded
            and str(d.get("abstract") or "").strip()]
    docs = [Counter(terms(str(d.get("title") or "") + " " + d["abstract"]))
            for d in pool]
    if not docs:
        return []
    average = sum(sum(d.values()) for d in docs) / len(docs)
    df = Counter(t for d in docs for t in d)
    q = sorted(set(terms(query)))
    scored = []
    for paper, bag in zip(pool, docs):
        score = 0.0
        dl = sum(bag.values())
        for t in q:
            freq = bag[t]
            if freq:
                idf = math.log(1 + (len(pool) - df[t] + .5) / (df[t] + .5))
                score += idf * freq * 2.2 / (freq + 1.2 * (.25 + .75 * dl / average))
        if score > 0:
            scored.append((score, paper))
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("paperId") or "")))
    return scored[:k]


def main():
    out = BASE / "review_packet"
    out.mkdir(parents=True, exist_ok=True)
    pairs, contracts, mapping, fingerprints, all_controls = [], [], [], {}, []
    workload = []
    for domain, run in RUNS.items():
        corpus = PROJECT / "runs" / run / "data/processed/corpus_filtered.jsonl"
        documents = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
        fingerprints[str(corpus.relative_to(PROJECT))] = digest(corpus)
        for path in sorted((ARCHIVE / domain).glob("cutoff_*/temporal_backtest.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            fingerprints[str(path.relative_to(PROJECT))] = digest(path)
            cutoff = int(data["cutoff_year"])
            local_pairs = 0
            for candidate in data["ranked_candidates"]:
                subject = str(candidate.get("subject") or candidate.get("head") or "")
                capability = str(candidate.get("missing_capability") or candidate.get("tail") or "")
                cid = "H-" + identifier([domain, cutoff, subject, capability])
                excluded = set(candidate.get("supporting_paper_ids") or [])
                query = subject + " " + capability
                hits = retrieve(documents, query, cutoff, excluded)
                contracts.append({"candidate_id": cid, "domain": domain,
                                  "subject": subject, "capability": capability,
                                  "contract": {"task": "", "domain": domain,
                                               "cutoff": f"{cutoff}-12-31",
                                               "conditions": [], "requirements": [],
                                               "scope_reviewed": False},
                                  "evidence": [],
                                  "ledger": {"cutoff": f"{cutoff}-12-31",
                                             "required_probe_ids": [], "probes": []}})
                origin_text = " | ".join(str(item.get("evidence") or "")
                                         for item in candidate.get("source_evidence", []))
                for rank, (score, paper) in enumerate(hits, 1):
                    pid = str(paper.get("paperId") or paper.get("paper_id") or "")
                    assert int(paper.get("year") or 0) <= cutoff and pid not in excluded
                    iid = "R-" + identifier([cid, pid])
                    pairs.append({"item_id": iid, "candidate_id": cid,
                                  "candidate_subject": subject, "candidate_capability": capability,
                                  "candidate_origin_span": origin_text,
                                  "cutoff": f"{cutoff}-12-31", "evidence_paper_id": pid,
                                  "evidence_title": paper.get("title", ""),
                                  "evidence_year": paper.get("year", ""),
                                  "evidence_doi": (paper.get("externalIds") or {}).get("DOI") or paper.get("doi", ""),
                                  "evidence_abstract": paper["abstract"]})
                    mapping.append({"item_id": iid, "domain": domain, "cutoff": cutoff,
                                    "candidate_id": cid, "rank": rank, "score": score,
                                    "source": "raw_title_abstract_BM25",
                                    "excluded_origin_ids": sorted(excluded)})
                    local_pairs += 1
            for control in data["positive_controls"]:
                all_controls.append({"domain": domain, "cutoff": cutoff, **control})
            workload.append({"domain": domain, "cutoff": cutoff,
                             "candidate_rows": len(data["ranked_candidates"]),
                             "review_pairs": local_pairs})
    ids = [p["item_id"] for p in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate review pairs")
    fields = list(pairs[0]) + ["scope_sufficient", "evidence_label", "source_span",
                              "experiment_id", "reviewer_id", "notes"]
    for reviewer in ("A", "B"):
        rows = list(pairs)
        random.Random(20261001 + ord(reviewer)).shuffle(rows)
        with (out / f"reviewer_{reviewer}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    (out / "scope_contracts_unannotated.json").write_text(
        json.dumps(contracts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "organizer_manifest.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "future_controls_for_audit.json").write_text(
        json.dumps(all_controls, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    unique_controls = {identifier([c["domain"], c["control_id"]]) for c in all_controls}
    report = {"status": "awaiting_independent_scope_annotations",
              "candidate_rows": len(contracts), "review_pairs": len(pairs),
              "unique_evidence_papers": len({p["evidence_paper_id"] for p in pairs}),
              "future_control_rows": len(all_controls),
              "distinct_future_control_ids_within_domain": len(unique_controls),
              "k": 5, "bm25": {"k1": 1.2, "b": .75},
              "interpretation": "Preparation/workload only; no scientific efficacy measured. Candidate generation is extraction-dependent; evidence ranking uses raw titles/abstracts only.",
              "origin_id_exclusion_and_recorded_year_checks_passed": True,
              "historical_index_availability_verified": False,
              "query": "pre-cutoff candidate subject + capability; no future controls",
              "human_labels_generated": 0,
              "workload": workload,
              "input_sha256": fingerprints,
              "script_sha256": digest(__file__)}
    (BASE / "scope_preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "candidate_rows", "review_pairs",
                                           "unique_evidence_papers", "human_labels_generated")}, indent=2))


if __name__ == "__main__":
    main()
