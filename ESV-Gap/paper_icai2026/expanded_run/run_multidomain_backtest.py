"""Offline four-domain rerun on preserved literature and triple snapshots.

This evaluates candidate ranking against heuristic future controls only. It does
not execute online retrieval, human review, full-text certification, or novelty
validation. Historical input directories are read-only; outputs stay here.

From ESV-Gap/: python -B paper_icai2026/expanded_run/run_multidomain_backtest.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import networkx
import yaml


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.temporal_backtest import run_cutoff_backtest, evaluate_ranked_candidates, tokens  # noqa: E402


OUTPUT = Path(__file__).resolve().parent
RUNS = {
    "iot_intrusion_detection": "deep_learning_iot_intrusion_de_20260831_114802",
    "microservice_security": "security_of_microservices_20260806_114632",
    "mongodb_security": "security_of_mongodb_20260806_135447",
    "handwritten_math_recognition": "handwritten_mathematical_expre_20260823_215228",
}
DOMAIN_TEXT = {
    "iot_intrusion_detection": "deep learning IoT intrusion detection",
    "microservice_security": "security of microservices",
    "mongodb_security": "security of MongoDB",
    "handwritten_math_recognition": "handwritten mathematical expression recognition",
}
CUTOFFS = (2022, 2023, 2024)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paper_key(doc: dict) -> str:
    doi = str((doc.get("externalIds") or {}).get("DOI") or doc.get("doi") or "").strip().lower()
    if doi:
        return "doi:" + doi.removeprefix("https://doi.org/")
    pid = str(doc.get("paperId") or doc.get("paper_id") or "").strip().lower()
    if pid:
        return "id:" + pid
    return "title:" + " ".join(str(doc.get("title") or "").lower().split())


def direct_limitation_baseline(triples: list[dict], pre_ids: set[str]) -> list[dict]:
    """Rank one direct LACKS candidate per normalized subject/capability pair.

    Uses the same frozen triples and no graph. The highest extractor confidence
    is retained; deterministic lexical tie-break avoids selection by future data.
    """
    by_pair = {}
    for triple in triples:
        if triple.get("source_paper_id") not in pre_ids or str(triple.get("relation") or "").upper() != "LACKS":
            continue
        subject = str((triple.get("subject") or {}).get("name") or "").strip()
        capability = str((triple.get("object") or {}).get("name") or "").strip()
        if not subject or not capability:
            continue
        key = (tuple(sorted(tokens(subject))), tuple(sorted(tokens(capability))))
        if not key[0] or not key[1]:
            continue
        value = {
            "type": "evidence_gap", "subject": subject,
            "missing_capability": capability,
            "mean_evidence_confidence": float(triple.get("confidence") or 0.0),
            "source_paper_id": triple["source_paper_id"],
        }
        old = by_pair.get(key)
        if old is None or (value["mean_evidence_confidence"], subject, capability) > (
            old["mean_evidence_confidence"], old["subject"], old["missing_capability"]
        ):
            by_pair[key] = value
    return sorted(by_pair.values(), key=lambda c: (-c["mean_evidence_confidence"], c["subject"], c["missing_capability"]))


def main() -> None:
    config_path = PROJECT / "config.yaml"
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summaries = []
    all_keys = []
    input_hashes = {}
    for label, run_name in RUNS.items():
        run = PROJECT / "runs" / run_name
        corpus_path = run / "data/processed/corpus_filtered.jsonl"
        triples_path = run / "data/triples/all_triples.json"
        documents = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        triples = json.loads(triples_path.read_text(encoding="utf-8"))
        keys = [paper_key(d) for d in documents]
        if any(k == "title:" for k in keys):
            raise ValueError(f"A source in {label} has no usable bibliographic identity")
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate source within {label}")
        all_keys.extend(keys)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["project"]["domain"] = DOMAIN_TEXT[label]
        years = Counter(int(d.get("year") or 0) for d in documents)
        domain_result = {
            "domain": label,
            "source_run": run_name,
            "retained_papers": len(documents),
            "triples": len(triples),
            "years": dict(sorted(years.items())),
            "cutoffs": [],
        }
        for cutoff in CUTOFFS:
            pre = sum(year <= cutoff for year in years.elements())
            future = len(documents) - pre
            if pre < 10 or future < 10:
                domain_result["cutoffs"].append({"cutoff": cutoff, "status": "insufficient_pre_or_future", "pre": pre, "future": future})
                continue
            result = run_cutoff_backtest(documents, triples, config, cutoff, OUTPUT / label)
            metrics = result["metrics"]
            pre_ids = {str(d.get("paperId") or d.get("paper_id") or "") for d in documents if int(d.get("year") or 0) <= cutoff}
            baseline = direct_limitation_baseline(triples, pre_ids)
            baseline_metrics = evaluate_ranked_candidates(baseline, result["positive_controls"], result["negative_controls"])
            domain_result["cutoffs"].append({
                "cutoff": cutoff,
                "status": "completed",
                "pre": result["pre_cutoff_papers"],
                "future": result["post_cutoff_papers"],
                "positive_controls": metrics["positive_control_count"],
                "negative_controls": metrics["negative_control_count"],
                "candidates": metrics["candidate_count"],
                "recall_at_5": metrics["candidate_recall_at_k"].get("5"),
                "recall_at_10": metrics["candidate_recall_at_k"].get("10"),
                "recall_at_20": metrics["candidate_recall_at_k"].get("20"),
                "direct_lacks_baseline_candidates": len(baseline),
                "direct_lacks_baseline_recall_at_20": baseline_metrics["candidate_recall_at_k"].get("20"),
                "unlabelled_candidates": metrics["unlabelled_candidate_count"],
                "certification_metric_status": metrics["certificate_metric_status"],
                "leakage_check_passed": result["leakage_check_passed"],
            })
        summaries.append(domain_result)
        input_hashes[label] = {
            "corpus": sha256(corpus_path),
            "triples": sha256(triples_path),
        }
    completed = [row for domain in summaries for row in domain["cutoffs"] if row["status"] == "completed"]
    scored = [row for row in completed if row["positive_controls"]]
    deltas = [row["recall_at_20"] - row["direct_lacks_baseline_recall_at_20"] for row in scored]
    report = {
        "status": "offline_retrospective_silver_control_replay",
        "interpretation": "Heuristic future controls are not expert-validated gaps. Certificate metrics are not estimated.",
        "source_rows": len(all_keys),
        "distinct_papers_by_doi_then_id": len(set(all_keys)),
        "cross_domain_duplicate_rows": len(all_keys) - len(set(all_keys)),
        "domain_count": len(summaries),
        "cutoff_protocol": list(CUTOFFS),
        "minimum_pre_and_future_papers": 10,
        "completed_domain_cutoffs": len(completed),
        "cutoffs_with_positive_controls": len(scored),
        "macro_recall_at_20_scored_cutoffs": round(sum(r["recall_at_20"] for r in scored) / len(scored), 4) if scored else None,
        "direct_lacks_baseline_macro_recall_at_20_scored_cutoffs": round(sum(r["direct_lacks_baseline_recall_at_20"] for r in scored) / len(scored), 4) if scored else None,
        "mean_paired_recall_at_20_delta_vs_direct_lacks": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "total_positive_controls_across_cutoffs_not_independent": sum(r["positive_controls"] for r in completed),
        "total_candidate_rows_across_cutoffs_not_independent": sum(r["candidates"] for r in completed),
        "all_implemented_support_id_checks_passed": all(r["leakage_check_passed"] for r in completed),
        "sources": summaries,
        "fingerprints": {
            "inputs": input_hashes,
            "config": sha256(config_path),
            "temporal_code": sha256(PROJECT / "src/temporal_backtest.py"),
            "detector_code": sha256(PROJECT / "src/detect_gaps.py"),
            "runner": sha256(Path(__file__)),
            "python": platform.python_version(),
            "networkx": networkx.__version__,
            "pyyaml": yaml.__version__,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "multidomain_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "fingerprints"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
