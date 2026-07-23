"""Run KG-TABI forward from the frozen lexical snapshot with stage manifests.

The runner deliberately starts from the released 185-record input snapshot;
it does not replay legacy retrieval or screening. It is resumable at extraction
through the checkpoint written by :mod:`src.extract_triples`. A non-empty run
ID is mandatory so no legacy diagnostic artifact is overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _configure_run_id_from_cli() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id", required=True)
    args, _ = parser.parse_known_args()
    os.environ["KG_TABI_RUN_ID"] = args.run_id


_configure_run_id_from_cli()

from src import config
from src.closure_search import build_closure_audit, build_closure_run_manifest, write_closure_outputs
from src.entity_resolution import run_resolution
from src.extract_triples import run_extraction
from src.graph_analysis import run_analysis
from src.provenance import file_sha256, utc_now_iso
from src.tabi_inference import run_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "runs" / "microservices-security-lexical-v1"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_frozen_input(snapshot_root: Path) -> dict[str, Any]:
    source_raw = snapshot_root / "raw_papers"
    required = ("papers_metadata.json", "screened_papers.json", "chunks.json", "corpus_manifest.json")
    missing = [name for name in required if not (source_raw / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen snapshot is incomplete: {', '.join(missing)}")
    target_raw = Path(config.RAW_PAPERS_DIR)
    target_raw.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in required:
        source = source_raw / name
        target = target_raw / name
        if target.exists() and file_sha256(target) != file_sha256(source):
            raise RuntimeError(
                f"Refusing to replace an existing nonmatching run input: {target}. "
                "Choose a fresh run ID."
            )
        if not target.exists():
            shutil.copy2(source, target)
        copied.append({
            "source": str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source_sha256": file_sha256(source),
            "target": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "target_sha256": file_sha256(target),
        })
    return {
        "source_snapshot": str(snapshot_root.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "snapshot_corpus_manifest_sha256": file_sha256(source_raw / "corpus_manifest.json"),
        "copied_artifacts": copied,
    }


def run_frozen_e2e(
    *, snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT, deterministic_closure_only: bool = True
) -> dict[str, Any]:
    """Execute the forward stages and return a non-secret run status record."""
    if not config.RUN_ID:
        raise RuntimeError("A non-empty --run-id is required for a forward E2E run.")
    manifest_path = Path(config.DATA_DIR) / "e2e_run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "frozen-e2e-run-v1",
        "run_id": config.RUN_ID,
        "started_at_utc": utc_now_iso(),
        "scope": (
            "Forward execution from a frozen lexical input. It is reproducible only to the "
            "extent that configured external LLM/API responses and model availability permit; "
            "it does not create human evaluation labels."
        ),
        "stages": {},
    }
    try:
        manifest["input"] = _copy_frozen_input(snapshot_root)
        manifest["stages"]["input_lock"] = "completed"
        triples = run_extraction()
        chunk_count = len(json.loads((Path(config.RAW_PAPERS_DIR) / "chunks.json").read_text(encoding="utf-8")))
        progress = json.loads((Path(config.TRIPLES_DIR) / "extraction_progress.json").read_text(encoding="utf-8"))
        completed_chunks = len(progress.get("processed_chunk_indices", []))
        manifest["stages"]["triple_extraction"] = {
            "status": "completed",
            "raw_triple_count": len(triples),
            "completed_chunks": completed_chunks,
            "expected_chunks": chunk_count,
            "raw_triples_sha256": file_sha256(Path(config.TRIPLES_DIR) / "raw_triples.json"),
        }
        if completed_chunks != chunk_count:
            raise RuntimeError(
                f"Extraction checkpoint is incomplete ({completed_chunks}/{chunk_count} chunks); "
                "rerun the same command after resolving failed provider calls."
            )
        if not triples:
            raise RuntimeError("Extraction returned zero valid triples; stopping before graph construction.")

        run_resolution()
        manifest["stages"]["entity_resolution"] = {
            "status": "completed",
            "manifest_sha256": file_sha256(Path(config.TRIPLES_DIR) / "entity_resolution_run_manifest.json"),
        }

        run_analysis()
        manifest["stages"]["topology_temporal"] = {
            "status": "completed",
            "topology_config_sha256": file_sha256(Path(config.GRAPH_DIR) / "topology_run_config.json"),
            "temporal_report_sha256": file_sha256(Path(config.GRAPH_DIR) / "temporal_screening_report.json"),
        }

        run_inference()
        gaps_path = Path(config.GAPS_DIR) / "kgtabi_gaps.json"
        gaps = json.loads(gaps_path.read_text(encoding="utf-8"))
        manifest["stages"]["tabi"] = {
            "status": "completed",
            "candidate_count": len(gaps),
            "output_sha256": file_sha256(gaps_path),
        }

        closure_audit_path = Path(config.GAPS_DIR) / "closure_search_audit.json"
        closure_manifest_path = Path(config.GAPS_DIR) / "closure_search_manifest.json"
        if gaps:
            audit = build_closure_audit(
                gaps,
                include_llm_variants=not deterministic_closure_only,
            )
            write_closure_outputs(
                audit,
                output_path=str(closure_audit_path),
                manifest_path=str(closure_manifest_path),
                input_path=str(gaps_path),
                limit=20,
                include_llm_variants=not deterministic_closure_only,
                citation_limit=10,
                citation_candidate_limit=5,
            )
            manifest["stages"]["closure_retrieval"] = {
                "status": "completed",
                "claim_count": len(audit),
                "deterministic_only": deterministic_closure_only,
                "manifest_sha256": file_sha256(closure_manifest_path),
            }
        else:
            _write_json(closure_audit_path, [])
            _write_json(closure_manifest_path, {
                "schema_version": "closure-retrieval-run-v1",
                "run_id": config.RUN_ID,
                "status": "not_invoked_no_tabi_candidates",
                "input_sha256": file_sha256(gaps_path),
                "generated_at_utc": utc_now_iso(),
            })
            manifest["stages"]["closure_retrieval"] = {
                "status": "not_invoked_no_tabi_candidates",
                "deterministic_only": deterministic_closure_only,
            }
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {"error_type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        manifest["completed_at_utc"] = utc_now_iso()
        _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KG-TABI from the frozen lexical snapshot.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument(
        "--closure-llm-variants",
        action="store_true",
        help="Enable supplemental LLM query variants when TABI creates candidates.",
    )
    args = parser.parse_args()
    report = run_frozen_e2e(
        snapshot_root=args.snapshot_root,
        deterministic_closure_only=not args.closure_llm_variants,
    )
    print(f"[+] Frozen E2E run completed: {report['status']} ({config.DATA_DIR})")


if __name__ == "__main__":
    main()
