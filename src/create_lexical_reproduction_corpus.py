"""Create a deterministic, auditable lexical corpus snapshot without API or LLM calls.

This utility does *not* turn the lexical screen into an expert relevance
judgment and does not run triple extraction.  It creates a stable input corpus
that can be used for a future end-to-end rerun after extraction provenance and
human screening are available.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from src.fetch_papers import chunk_text, screen_relevance_lexical


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "runs"
    / "microservices-security-v1"
    / "raw_papers"
    / "papers_metadata.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "runs" / "microservices-security-lexical-v1" / "raw_papers"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_lexical_corpus(source_path: Path, output_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Create an auditable 185-record lexical snapshot from a retrieved-record snapshot."""
    if not source_path.exists():
        raise FileNotFoundError(f"Retrieved-record snapshot does not exist: {source_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}. Use --overwrite to replace it."
        )

    source_bytes = source_path.read_bytes()
    retrieved = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(retrieved, list):
        raise ValueError("The retrieved-record snapshot must be a JSON list.")

    # The screening function intentionally annotates input records, so copy the
    # source snapshot first.  This guarantees that a lexical decision is
    # derived solely from the frozen 314-record input.
    screened = screen_relevance_lexical(copy.deepcopy(retrieved), max_papers=len(retrieved))

    chunks: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    retained_ids = {
        str(paper.get("paperId", paper.get("paper_id", "")))
        for paper in screened
    }
    for source_index, paper in enumerate(retrieved):
        paper_id = str(paper.get("paperId", paper.get("paper_id", "")))
        decisions.append(
            {
                "source_index": source_index,
                "paper_id": paper_id,
                "included_by_lexical_rule": paper_id in retained_ids,
            }
        )

    for screened_rank, paper in enumerate(screened):
        paper_id = str(paper.get("paperId", paper.get("paper_id", "")))
        for chunk_index, text in enumerate(chunk_text(paper.get("abstract", ""), max_words=1000)):
            chunks.append(
                {
                    "paperId": paper_id,
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "chunk_index": chunk_index,
                    "section_label": "abstract",
                    "screened_rank": screened_rank,
                    "text": text,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "papers_metadata.json", retrieved)
    _write_json(output_dir / "screened_papers.json", screened)
    _write_json(output_dir / "chunks.json", chunks)

    with (output_dir / "screening_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]) if decisions else ["source_index", "paper_id", "included_by_lexical_rule"])
        writer.writeheader()
        writer.writerows(decisions)

    rule_source = inspect.getsource(screen_relevance_lexical).encode("utf-8")
    manifest = {
        "schema_version": "lexical-reproduction-corpus-v1",
        "status": (
            "Deterministic lexical corpus snapshot only; not a human-adjudicated "
            "corpus and not an end-to-end KG-TABI result."
        ),
        "source_snapshot": {
            "path": str(source_path),
            "sha256": _sha256_bytes(source_bytes),
            "retrieved_records": len(retrieved),
        },
        "screening": {
            "function": "src.fetch_papers.screen_relevance_lexical",
            "function_sha256": _sha256_bytes(rule_source),
            "input": "lower-cased title concatenated with abstract",
            "include_rule": "at least one architecture term AND at least one security term",
            "retained_records": len(screened),
            "decision_file": "screening_decisions.csv",
        },
        "text_scope": {
            "chunks": len(chunks),
            "section_label": "abstract",
            "full_text_chunks": 0,
        },
        "not_performed": [
            "human screening adjudication",
            "triple extraction",
            "entity resolution",
            "graph construction",
            "TABI generation",
        ],
    }
    _write_json(output_dir / "corpus_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic lexical corpus snapshot from a frozen retrieval snapshot."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = build_lexical_corpus(args.source, args.output_dir, overwrite=args.overwrite)
    print(
        "[+] Wrote reproducible lexical snapshot: "
        f"{manifest['screening']['retained_records']} retained / "
        f"{manifest['source_snapshot']['retrieved_records']} retrieved records to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
