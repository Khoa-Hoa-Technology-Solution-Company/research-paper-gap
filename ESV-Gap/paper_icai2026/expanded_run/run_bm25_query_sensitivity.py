"""Post-hoc query sensitivity for the extraction-dependent BM25-LACKS ranking.

From ESV-Gap/: python -B paper_icai2026/expanded_run/run_bm25_query_sensitivity.py
The three templates are descriptive robustness probes, not preregistered choices.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_icai2026.expanded_run.run_bm25_lacks_baseline import (
    BASE, PROJECT, RUNS, DOMAIN_TEXT, bm25_scores, digest, matched_ids,
    direct_limitation_baseline, evaluate_ranked_candidates,
)


QUERIES = {
    "Q1": "limitation unresolved challenge",
    "Q2": "research gap limitation problem",
    "Q3": "limitation future work challenge",
}


def main() -> None:
    rows = []
    inputs = {}
    for domain, run_name in RUNS.items():
        corpus = PROJECT / "runs" / run_name / "data/processed/corpus_filtered.jsonl"
        triples_path = PROJECT / "runs" / run_name / "data/triples/all_triples.json"
        documents = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
        triples = json.loads(triples_path.read_text(encoding="utf-8"))
        inputs[domain] = {"corpus_sha256": digest(corpus), "triples_sha256": digest(triples_path)}
        for saved_path in sorted((BASE / domain).glob("cutoff_*/temporal_backtest.json")):
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            cutoff = int(saved["cutoff_year"])
            pre_ids = {str(doc.get("paperId") or doc.get("paper_id") or "")
                       for doc in documents if int(doc.get("year") or 0) <= cutoff}
            direct = direct_limitation_baseline(triples, pre_ids)
            positives = saved["positive_controls"]
            negatives = saved["negative_controls"]
            result = {"domain": domain, "cutoff": cutoff, "positive_controls": len(positives),
                      "candidate_count": len(direct), "saved_replay_sha256": digest(saved_path),
                      "queries": {}}
            for name, suffix in QUERIES.items():
                query = DOMAIN_TEXT[domain] + " " + suffix
                scores = bm25_scores([str(c["subject"]) + " " + str(c["missing_capability"])
                                      for c in direct], query)
                ranked = sorted(zip(direct, scores), key=lambda pair: (
                    -pair[1], -pair[0]["mean_evidence_confidence"],
                    pair[0]["subject"], pair[0]["missing_capability"],
                ))
                candidates = [{**candidate, "candidate_quality": {"score": score}}
                              for candidate, score in ranked]
                metrics = evaluate_ranked_candidates(candidates, positives, negatives)
                result["queries"][name] = {
                    "query": query,
                    "recall_at_k": metrics["candidate_recall_at_k"],
                    "matched_control_ids_at_20": sorted(matched_ids(candidates, positives)),
                }
            rows.append(result)
    scored = [row for row in rows if row["positive_controls"]]
    if len(rows) != 10 or len(scored) != 9:
        raise ValueError("Unexpected evaluation count")
    summary = {}
    for name in QUERIES:
        summary[name] = {
            "macro_recall_at_k": {
                str(k): round(sum(row["queries"][name]["recall_at_k"][str(k)]
                                  for row in scored) / len(scored), 4)
                for k in (5, 10, 20)
            },
            "matched_control_rows_at_20_not_independent": sum(
                len(row["queries"][name]["matched_control_ids_at_20"]) for row in scored
            ),
        }
    report = {
        "status": "post_hoc_query_sensitivity_not_preregistered",
        "interpretation": "All queries rerank the same pre-cutoff extracted LACKS pool; controls remain heuristic.",
        "query_suffixes": QUERIES,
        "completed_evaluations": len(rows),
        "scored_evaluations": len(scored),
        "summary": summary,
        "rows": rows,
        "input_hashes": inputs,
        "script_sha256": digest(Path(__file__)),
    }
    target = BASE / "bm25_query_sensitivity.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
