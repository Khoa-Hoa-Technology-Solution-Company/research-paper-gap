"""Audit provider-returned model identifiers in a completed forward run.

The provider may route one configured model string to multiple reported model
identifiers.  This utility quantifies that observable fact and links it to
per-call triple counts, without claiming a causal model-quality comparison:
different calls contain different abstracts and there are no repeated inputs.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _configure_run_id_from_cli() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id")
    args, _ = parser.parse_known_args()
    if args.run_id:
        os.environ["KG_TABI_RUN_ID"] = args.run_id


_configure_run_id_from_cli()

from src import config
from src.provenance import file_sha256, utc_now_iso


def build_provider_model_audit(
    calls: list[dict[str, Any]], triples: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return a descriptive, non-causal routing and output-count report."""
    extraction_calls = [
        call for call in calls
        if call.get("purpose") == "typed-triple-extraction"
    ]
    triples_by_call: dict[str, int] = Counter(
        str(triple.get("extraction_call_id") or "missing-call-id")
        for triple in triples
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in extraction_calls:
        response = call.get("response") or {}
        identifier = str(response.get("provider_model_identifier") or "missing-provider-identifier")
        grouped[identifier].append(call)

    rows = []
    for identifier, records in sorted(grouped.items()):
        counts = [triples_by_call[str(record.get("call_id"))] for record in records]
        rows.append({
            "provider_model_identifier": identifier,
            "calls": len(records),
            "share_of_extraction_calls": len(records) / len(extraction_calls) if extraction_calls else 0.0,
            "triples": sum(counts),
            "triples_per_call_mean": sum(counts) / len(counts) if counts else 0.0,
            "triples_per_call_median": median(counts) if counts else 0.0,
            "triples_per_call_min": min(counts) if counts else 0,
            "triples_per_call_max": max(counts) if counts else 0,
        })

    retry_counts = [len(call.get("attempts") or []) for call in extraction_calls]
    configured_models = Counter(
        str((call.get("llm") or {}).get("model_identifier") or "missing-configured-model")
        for call in extraction_calls
    )
    statuses = Counter(str(call.get("status") or "missing-status") for call in extraction_calls)
    return {
        "schema_version": "provider-model-audit-v1",
        "generated_at_utc": utc_now_iso(),
        "scope": (
            "Descriptive audit of provider-returned metadata and triple counts. It does not "
            "identify the underlying model weights, establish deterministic routing, or estimate "
            "a causal effect of identifier on extraction quality."
        ),
        "configured_model_identifiers": dict(sorted(configured_models.items())),
        "provider_model_identifier_distribution": rows,
        "call_statuses": dict(sorted(statuses.items())),
        "retry_summary": {
            "calls_with_retry": sum(count > 1 for count in retry_counts),
            "maximum_attempts_observed": max(retry_counts, default=0),
        },
        "trace_fields_retained": [
            "configured provider/model/base URL", "provider model identifier", "provider response ID",
            "provider system fingerprint when supplied", "prompt hash/version", "generation settings",
            "retry attempts", "raw response hash", "assistant-content hash", "extraction call ID",
        ],
        "raw_response_policy": (
            (extraction_calls[0].get("raw_response_storage_policy") if extraction_calls else None)
        ),
        "interpretation": (
            "Multiple provider identifiers mean the run is provenance-complete only to the granularity "
            "the provider exposes. Re-running cannot guarantee identical underlying routing or outputs; "
            "the observed triple-count differences are confounded by differing input abstracts."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit provider model identifiers in a forward run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(config.DATA_DIR) / "provider_model_audit.json"
    )
    args = parser.parse_args()
    if not config.RUN_ID:
        parser.error("--run-id is required")
    call_path = Path(config.PROVENANCE_DIR) / "llm_calls.jsonl"
    triples_path = Path(config.TRIPLES_DIR) / "raw_triples.json"
    calls = [json.loads(line) for line in call_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    triples = json.loads(triples_path.read_text(encoding="utf-8"))
    report = build_provider_model_audit(calls, triples)
    report["input_hashes"] = {
        "llm_calls_jsonl": file_sha256(call_path),
        "raw_triples_json": file_sha256(triples_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Wrote provider model audit to {args.output}")


if __name__ == "__main__":
    main()
