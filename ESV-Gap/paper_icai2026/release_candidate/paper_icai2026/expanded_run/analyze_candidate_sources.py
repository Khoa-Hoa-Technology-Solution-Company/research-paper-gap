"""Recount saved candidate sources without generating labels or rerunning extraction.

From ESV-Gap/: python -B paper_icai2026/expanded_run/analyze_candidate_sources.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.temporal_backtest import evaluate_ranked_candidates  # noqa: E402


BASE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    for path in sorted(BASE.glob("*/cutoff_*/temporal_backtest.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        candidates = saved["ranked_candidates"]
        explicit = [c for c in candidates if c.get("type") == "evidence_gap"]
        map_cells = [c for c in candidates if c.get("detector") == "typed_evidence_map_empty_cell"]
        other = [c for c in candidates if c not in explicit and c not in map_cells]
        if other or len(explicit) + len(map_cells) != len(candidates):
            raise ValueError(f"Unexpected or overlapping candidate type: {path}")
        positives = saved["positive_controls"]
        negatives = saved["negative_controls"]
        explicit_metrics = evaluate_ranked_candidates(explicit, positives, negatives)
        map_metrics = evaluate_ranked_candidates(map_cells, positives, negatives)
        original_recall = saved["metrics"]["candidate_recall_at_k"]["20"]
        union_metrics = evaluate_ranked_candidates(candidates, positives, negatives)
        if union_metrics["candidate_recall_at_k"]["20"] != original_recall:
            raise ValueError(f"Saved replay recall changed: {path}")
        rows.append({
            "domain": path.parent.parent.name,
            "cutoff": saved["cutoff_year"],
            "positive_controls": len(positives),
            "candidate_count": len(candidates),
            "explicit_limitations": len(explicit),
            "typed_evidence_map_cells": len(map_cells),
            "recall_at_20_union": original_recall,
            "recall_at_20_explicit": explicit_metrics["candidate_recall_at_k"]["20"],
            "recall_at_20_map": map_metrics["candidate_recall_at_k"]["20"],
            "saved_replay_sha256": digest(path),
        })
    if len(rows) != 10:
        raise ValueError(f"Expected ten completed evaluations, found {len(rows)}")
    output = {
        "status": "offline_source_ablation_on_saved_candidates_and_heuristic_controls",
        "interpretation": "Generator contribution to proxy retrieval, not causal graph benefit or validated gap discovery.",
        "evaluations": len(rows),
        "candidate_rows_not_independent": sum(row["candidate_count"] for row in rows),
        "explicit_limitation_rows_not_independent": sum(row["explicit_limitations"] for row in rows),
        "typed_evidence_map_rows_not_independent": sum(row["typed_evidence_map_cells"] for row in rows),
        "rows": rows,
        "script_sha256": digest(Path(__file__)),
    }
    target = BASE / "candidate_source_ablation.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "evaluations", "candidate_rows_not_independent",
        "explicit_limitation_rows_not_independent", "typed_evidence_map_rows_not_independent"
    )}, indent=2))


if __name__ == "__main__":
    main()
