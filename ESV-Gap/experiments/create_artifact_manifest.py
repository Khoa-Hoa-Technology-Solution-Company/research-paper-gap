"""Hash the exact code, configuration, reports, and manuscript sources."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "config.yaml",
    "requirements.txt",
    "requirements-experiments.lock",
    "app.py",
    "prompts/filter_abstract.txt",
    "src/collect.py",
    "src/filter.py",
    "src/extract_triples.py",
    "src/rag_baseline.py",
    "src/validate_gaps.py",
    "src/entity_normalization.py",
    "src/build_graph.py",
    "src/detect_gaps.py",
    "src/score_gaps.py",
    "tests/test_validate_gaps.py",
    "tests/test_collect.py",
    "tests/test_build_graph.py",
    "tests/test_llm_fallbacks.py",
    "experiments/run_validation_benchmark.py",
    "experiments/run_corpus_validation.py",
    "experiments/prepare_expert_review.py",
    "experiments/audit_corpus_integrity.py",
    "experiments/create_artifact_manifest.py",
    "outputs/validation_benchmark_v3.json",
    "outputs/microservices_corpus_validation_v3.json",
    "outputs/expert_review_packet.csv",
    "outputs/expert_review_manifest.json",
    "outputs/corpus_integrity_audit.json",
    "runs/reseach_gap_20260806_092230/data/processed/corpus_filtered.jsonl",
    "runs/reseach_gap_20260806_092230/data/triples/all_triples.json",
    "runs/reseach_gap_20260806_092230/outputs/detected_gaps_raw.json",
    "runs/reseach_gap_20260806_092230/outputs/gaps_ranked_top.json",
    "paper_v2/main.tex",
    "paper_v2/main.pdf",
    "paper_v2/references.bib",
    "paper_v2/results_summary.json",
    "paper_v2/RESPONSE_TO_REVIEWERS.md",
    "paper_v2/REPRODUCIBILITY.md",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "base_git_commit": commit,
        "working_tree_revision": True,
        "files": {name: sha256(ROOT / name) for name in FILES},
    }
    canonical = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":"))
    manifest["patchset_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    output = ROOT / "outputs" / "artifact_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
