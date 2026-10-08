"""Offline BM25 reranking of the same direct-LACKS candidate pool.

This is an extraction-dependent lexical ranking comparator, not a full
text-only research-gap discovery pipeline. Queries are fixed from each
domain label plus generic limitation terms; no future controls form queries.

From ESV-Gap/: python -B paper_icai2026/expanded_run/run_bm25_lacks_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from paper_icai2026.expanded_run.run_multidomain_backtest import (  # noqa: E402
    DOMAIN_TEXT, RUNS, direct_limitation_baseline,
)
from src.temporal_backtest import (  # noqa: E402
    candidate_matches_control, evaluate_ranked_candidates,
)


BASE = Path(__file__).resolve().parent
EXTRA_QUERY = "limitation unresolved challenge"
K1 = 1.2
B = 0.75


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold().replace("-", " "))


def bm25_scores(texts: list[str], query: str) -> list[float]:
    docs = [Counter(terms(text)) for text in texts]
    n = len(docs)
    if not n:
        return []
    avg_len = sum(sum(doc.values()) for doc in docs) / n
    query_terms = set(terms(query))
    doc_freq = Counter(term for doc in docs for term in doc)
    scores = []
    for doc in docs:
        dl = sum(doc.values())
        score = 0.0
        for term in query_terms:
            freq = doc[term]
            if not freq:
                continue
            idf = math.log(1.0 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += idf * freq * (K1 + 1.0) / (
                freq + K1 * (1.0 - B + B * dl / avg_len)
            )
        scores.append(score)
    return scores


def matched_ids(candidates: list[dict], controls: list[dict]) -> set[str]:
    return {
        control["control_id"]
        for candidate in candidates[:20]
        for control in controls
        if candidate_matches_control(candidate, control)
    }


def main() -> None:
    rows = []
    input_hashes = {}
    for domain, run_name in RUNS.items():
        corpus_path = PROJECT / "runs" / run_name / "data/processed/corpus_filtered.jsonl"
        triples_path = PROJECT / "runs" / run_name / "data/triples/all_triples.json"
        documents = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        triples = json.loads(triples_path.read_text(encoding="utf-8"))
        input_hashes[domain] = {"corpus": digest(corpus_path), "triples": digest(triples_path)}
        for cutoff_path in sorted((BASE / domain).glob("cutoff_*/temporal_backtest.json")):
            saved = json.loads(cutoff_path.read_text(encoding="utf-8"))
            cutoff = int(saved["cutoff_year"])
            pre_ids = {
                str(doc.get("paperId") or doc.get("paper_id") or "")
                for doc in documents if int(doc.get("year") or 0) <= cutoff
            }
            direct = direct_limitation_baseline(triples, pre_ids)
            query = DOMAIN_TEXT[domain] + " " + EXTRA_QUERY
            scores = bm25_scores([
                str(candidate["subject"]) + " " + str(candidate["missing_capability"])
                for candidate in direct
            ], query)
            ranked = sorted(
                zip(direct, scores),
                key=lambda pair: (
                    -pair[1], -pair[0]["mean_evidence_confidence"],
                    pair[0]["subject"], pair[0]["missing_capability"],
                ),
            )
            bm25 = [{
                **candidate,
                "candidate_quality": {"score": score},
            } for candidate, score in ranked]
            positives = saved["positive_controls"]
            negatives = saved["negative_controls"]
            metrics = evaluate_ranked_candidates(bm25, positives, negatives)
            confidence_metrics = evaluate_ranked_candidates(direct, positives, negatives)
            if confidence_metrics["candidate_recall_at_k"]["20"] != saved.get(
                "direct_lacks_baseline_recall_at_20", confidence_metrics["candidate_recall_at_k"]["20"]
            ):
                raise ValueError(f"Confidence comparator mismatch: {cutoff_path}")
            rows.append({
                "domain": domain,
                "cutoff": cutoff,
                "query": query,
                "positive_controls": len(positives),
                "candidate_count": len(bm25),
                "recall_at_k": metrics["candidate_recall_at_k"],
                "confidence_lacks_recall_at_k": confidence_metrics["candidate_recall_at_k"],
                "esv_gap_recall_at_k": saved["metrics"]["candidate_recall_at_k"],
                "matched_control_ids_at_20": sorted(matched_ids(bm25, positives)),
                "matched_pairs_at_20": [
                    {
                        "rank": rank,
                        "candidate_subject": candidate["subject"],
                        "candidate_capability": candidate["missing_capability"],
                        "bm25_score": candidate["candidate_quality"]["score"],
                        "control_id": control["control_id"],
                        "control_subject": control.get("subject"),
                        "control_capability": control.get("missing_capability"),
                        "future_paper_id": control.get("paper_id"),
                    }
                    for rank, candidate in enumerate(bm25[:20], start=1)
                    for control in positives
                    if candidate_matches_control(candidate, control)
                ],
                "saved_replay_sha256": digest(cutoff_path),
            })
    scored = [row for row in rows if row["positive_controls"]]
    if len(rows) != 10 or len(scored) != 9:
        raise ValueError("Unexpected saved evaluation count")
    report = {
        "status": "offline_extraction_dependent_bm25_lacks_reranking",
        "interpretation": "BM25 ranks pre-cutoff direct-LACKS subject/capability text, not raw abstracts; generated heuristic controls are not expert labels.",
        "query_protocol": "Domain label plus fixed generic terms: limitation unresolved challenge. No future-control text is used in ranking.",
        "bm25_parameters": {"k1": K1, "b": B, "idf": "log(1+(N-df+0.5)/(df+0.5))"},
        "completed_evaluations": len(rows),
        "scored_evaluations": len(scored),
        "macro_recall_at_k": {
            str(k): round(sum(row["recall_at_k"][str(k)] for row in scored) / len(scored), 4)
            for k in (5, 10, 20)
        },
        "macro_confidence_lacks_recall_at_k": {
            str(k): round(sum(row["confidence_lacks_recall_at_k"][str(k)] for row in scored) / len(scored), 4)
            for k in (5, 10, 20)
        },
        "macro_esv_gap_recall_at_k": {
            str(k): round(sum(row["esv_gap_recall_at_k"][str(k)] for row in scored) / len(scored), 4)
            for k in (5, 10, 20)
        },
        "matched_control_rows_at_20_not_independent": sum(len(row["matched_control_ids_at_20"]) for row in scored),
        "positive_control_rows_not_independent": sum(row["positive_controls"] for row in scored),
        "rows": rows,
        "input_hashes": input_hashes,
        "script_sha256": digest(Path(__file__)),
    }
    target = BASE / "bm25_lacks_baseline.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "completed_evaluations", "scored_evaluations", "macro_recall_at_k",
        "matched_control_rows_at_20_not_independent",
        "positive_control_rows_not_independent",
    )}, indent=2))


if __name__ == "__main__":
    main()
