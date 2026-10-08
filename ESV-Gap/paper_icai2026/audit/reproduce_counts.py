"""Recount immutable IoT run artifacts using Python's standard library only.

From ESV-Gap/: python -B paper_icai2026/audit/reproduce_counts.py
Writes only audit/reproduced_counts.json. Does not replay the research pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / "runs" / "deep_learning_iot_intrusion_de_20260831_114802"
OUT = Path(__file__).resolve().parent / "reproduced_counts.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(relative: str) -> dict:
    return json.loads((RUN / relative).read_text(encoding="utf-8"))


def counts_from_temporal(report: dict) -> list[dict]:
    return [
        {
            "cutoff": item["cutoff_year"],
            "pre_papers": item["pre_cutoff_papers"],
            "future_papers": item["post_cutoff_papers"],
            "positive_controls": item["metrics"]["positive_control_count"],
            "negative_controls": item["metrics"]["negative_control_count"],
            "candidates": item["metrics"]["candidate_count"],
            "recall_at_20": item["metrics"]["candidate_recall_at_k"]["20"],
            "unlabelled_candidates": item["metrics"]["unlabelled_candidate_count"],
            "certified_candidates": item["metrics"]["certified_candidate_count"],
        }
        for item in report["cutoff_results"]
    ]


def main() -> None:
    corpus = RUN / "data/processed/corpus_filtered.jsonl"
    triples_path = RUN / "data/triples/all_triples.json"
    raw_path = RUN / "outputs/detected_gaps_raw.json"
    audit_path = RUN / "outputs/gap_validation_audit.json"
    temporal_path = RUN / "outputs/temporal_backtest/temporal_backtest_summary.json"
    pdf_root = PROJECT.parent / "ESV-gap submit AMI.pdf"
    pdf_ami = PROJECT / "paper_v2/main_springer_ami.pdf"
    corpus_lines = [line for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in corpus_lines:
        json.loads(line)
    triples = json.loads(triples_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    diagnostic = read_json("data/processed/screening_diagnostics.json")
    full_text = read_json("outputs/full_text_enrichment.json")
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    signal_types = {key: len(value) for key, value in raw.items()}
    dispositions = {key: audit["summary"][key] for key in ("automatically_eligible", "review_required", "rejected")}
    assert sum(signal_types.values()) == audit["summary"]["raw_candidates"] == sum(dispositions.values())
    assert len(corpus_lines) == diagnostic["retained_paper_count"]
    assert len(triples) == 1404
    result = {
        "scope": "Read-only recount of frozen artifacts; saved temporal report is not recomputed by this script.",
        "source_run": RUN.name,
        "corpus": {
            "requested_collection_pool": read_json("run_metadata.json")["collection_pool_size"],
            "raw_records_observed": diagnostic["raw_paper_count"],
            "screened_records": diagnostic["screened_paper_count"],
            "retained_papers": len(corpus_lines),
            "relation_events": len(triples),
        },
        "raw_signals": signal_types,
        "raw_signal_total": sum(signal_types.values()),
        "validation_dispositions": dispositions,
        "full_text_enrichment": {key: full_text[key] for key in ("target_paper_count", "attempted_download_count", "enriched")},
        "saved_temporal_snapshot": {
            "aggregate": temporal["aggregate"],
            "cutoffs": counts_from_temporal(temporal),
        },
        "pdf_identity": {
            "root_pdf_sha256": digest(pdf_root),
            "springer_ami_pdf_sha256": digest(pdf_ami),
            "identical": digest(pdf_root) == digest(pdf_ami),
        },
        "input_sha256": {str(path.relative_to(PROJECT)): digest(path) for path in (corpus, triples_path, raw_path, audit_path, temporal_path)},
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
