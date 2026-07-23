"""Create a content-addressed manifest for a KG-TABI artifact snapshot.

The manifest intentionally distinguishes a Git commit from a dirty working
tree.  A hash over the listed inputs, outputs, and selected source files gives
an inspectable artifact identifier even before a maintainer creates a tagged,
immutable release.  It does not claim that a live API or an LLM provider can
be replayed identically.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _configure_run_id_from_cli() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id", default=None)
    args, _ = parser.parse_known_args()
    if args.run_id:
        os.environ["KG_TABI_RUN_ID"] = args.run_id


_configure_run_id_from_cli()

from src import config
from src.provenance import canonical_json, file_sha256, sha256_text, utc_now_iso


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_VERSION = "kgtabi-artifact-snapshot-v1"
SOURCE_PATHS = (
    "src/config.py",
    "src/fetch_papers.py",
    "src/extract_triples.py",
    "src/entity_resolution.py",
    "src/entity_resolution_diagnostics.py",
    "src/graph_analysis.py",
    "src/compatibility.py",
    "src/tabi_inference.py",
    "src/closure_search.py",
    "src/llm_client.py",
    "src/provenance.py",
    "src/traceability_sensitivity.py",
    "src/forward_sensitivity_diagnostics.py",
    "src/provider_model_audit.py",
    "src/temporal_benchmark.py",
    "src/run_frozen_e2e.py",
    "src/prepare_candidate_blind_review.py",
    "src/controlled_benchmark.py",
    "src/benchmark_reporting.py",
    "paper.tex",
    "paper.pdf",
    "supplementary.tex",
    "supplementary.pdf",
    ".gitignore",
    "references.bib",
    "README.md",
    "CITATION.cff",
    "requirements.txt",
    "docs/PROVENANCE.md",
    "docs/ARTIFACT_SCOPE.md",
    "docs/AUDIT_PACKET_AND_TRACE_CHAIN.md",
    "docs/IMMUTABLE_RELEASE.md",
    "docs/SUBMISSION_READINESS.md",
    "docs/RELEASE_NOTES_DRAFT.md",
    "docs/ANNOTATION_PROTOCOL.md",
    "docs/BLINDED_CANDIDATE_EVALUATION.md",
    "docs/REVIEW_RESPONSE.md",
)
ARTIFACT_PATHS = (
    "e2e_run_manifest.json",
    "raw_papers/papers_metadata.json",
    "raw_papers/screened_papers.json",
    "raw_papers/chunks.json",
    "triples/raw_triples.json",
    "triples/resolved_triples.json",
    "triples/entity_mapping.json",
    "triples/extraction_provenance_manifest.json",
    "triples/entity_resolution_run_manifest.json",
    "entity_resolution_legacy_diagnostic.json",
    "cut_edge_reconciliation.json",
    "graph/knowledge_graph.gml",
    "graph/orphan_clusters.json",
    "graph/topology_run_config.json",
    "graph/temporal_screening_report.json",
    "graph/temporal_decay.json",
    "gaps/kgtabi_gaps.json",
    "gaps/closure_search_audit.json",
    "gaps/closure_search_manifest.json",
    "provenance/llm_calls.jsonl",
    "louvain_robustness_results.json",
    "sensitivity_results.json",
    "traceability_restricted_sensitivity.json",
    "forward_sensitivity_diagnostic.json",
    "provider_model_audit.json",
    "temporal_signal_injection.json",
    "controlled_benchmark_multicondition_results.json",
    "controlled_benchmark_multicondition_trials.jsonl",
    "controlled_benchmark_appendix_report.json",
)
PACKAGE_NAMES = (
    "networkx",
    "python-louvain",
    "numpy",
    "rapidfuzz",
    "sentence-transformers",
    "requests",
    "openai",
)


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _path_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_manifest(*, run_root: Path | None = None) -> dict[str, Any]:
    """Build a manifest without writing it, making it easy to unit-test."""
    active_run_root = run_root or Path(config.DATA_DIR)
    source_records = [
        _path_record(PROJECT_ROOT / relative, relative_to=PROJECT_ROOT)
        for relative in SOURCE_PATHS
        if (PROJECT_ROOT / relative).is_file()
    ]
    artifact_records = [
        _path_record(active_run_root / relative, relative_to=active_run_root)
        for relative in ARTIFACT_PATHS
        if (active_run_root / relative).is_file()
    ]
    status = _git_output("status", "--porcelain=v1")
    git_state = {
        "head_commit": _git_output("rev-parse", "HEAD"),
        "working_tree_status_sha256": sha256_text(status) if status is not None else None,
        "working_tree_clean": status == "" if status is not None else None,
        "note": (
            "A nonempty status means this is a content-addressed working-tree snapshot, "
            "not an immutable tagged Git release."
        ),
    }
    identity_payload = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": config.RUN_ID or None,
        "source_files": source_records,
        "artifact_files": artifact_records,
        "git": git_state,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    return {
        **identity_payload,
        "created_at_utc": utc_now_iso(),
        "artifact_snapshot_sha256": sha256_text(canonical_json(identity_payload)),
        "scope": (
            "Content-addressed code-and-artifact inventory. It identifies this local snapshot "
            "but does not make mutable APIs, model providers, legacy screening, or legacy LLM "
            "calls end-to-end reproducible."
        ),
        "next_release_step": (
            "Create a reviewed Git commit and immutable archival release containing this manifest "
            "before citing it as a versioned public artifact."
        ),
    }


def write_manifest(output_path: Path, *, run_root: Path | None = None) -> dict[str, Any]:
    manifest = build_manifest(run_root=run_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a content-addressed KG-TABI artifact manifest.")
    parser.add_argument("--run-id", default=config.RUN_ID or None)
    parser.add_argument("--output", type=Path, default=Path(config.DATA_DIR) / "reproducibility_manifest.json")
    args = parser.parse_args()
    if args.run_id and args.run_id != config.RUN_ID:
        raise ValueError("--run-id must be supplied before import; use python -m ... --run-id <id>.")
    manifest = write_manifest(args.output)
    print(f"[+] Wrote artifact snapshot {manifest['artifact_snapshot_sha256']} to {args.output}")


if __name__ == "__main__":
    main()
