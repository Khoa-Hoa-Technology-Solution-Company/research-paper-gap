"""Auditable candidate retrieval for checking whether a claim may be addressed.

Closure retrieval is deliberately a *candidate discovery* step.  It does not
decide novelty, importance, or whether any source supports a hypothesis.  A
future rerun writes enough retrieval and query-generation provenance to audit
the candidate set, including deterministic variants that do not depend on an
LLM.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

import requests

from src import config
from src.llm_client import LLMCallError, call_llm_with_provenance, configured_llm_identity
from src.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    file_sha256,
    normalized_query_key,
    sha256_json,
    sha256_text,
    utc_now_iso,
)


FIELDS = "paperId,title,abstract,year,citationCount,externalIds,url"
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "should", "that", "the", "their",
    "this", "to", "using", "with", "would", "research", "study", "studies", "approach",
}


def _headers() -> dict[str, str]:
    return {"x-api-key": config.SEMANTIC_SCHOLAR_API_KEY} if config.SEMANTIC_SCHOLAR_API_KEY else {}


def _sleep_between_requests() -> None:
    if config.SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS:
        time.sleep(config.SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS)


def _request_manifest(
    *, endpoint: str, parameters: dict[str, Any], started_at_utc: str
) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "request_parameters": parameters,
        "requested_at_utc": started_at_utc,
        "timeout_seconds": config.SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS,
        "api_key_configured": bool(config.SEMANTIC_SCHOLAR_API_KEY),
    }


def search_candidate_prior_work_with_manifest(
    claim_query: str, limit: int = 20
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search Semantic Scholar and retain parameters, response hash, and IDs."""
    endpoint = f"{config.SEMANTIC_SCHOLAR_API_BASE_URL}/paper/search"
    parameters = {"query": claim_query, "limit": limit, "fields": FIELDS}
    request = _request_manifest(
        endpoint=endpoint, parameters=parameters, started_at_utc=utc_now_iso()
    )
    try:
        _sleep_between_requests()
        response = requests.get(
            endpoint,
            params=parameters,
            headers=_headers(),
            timeout=config.SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS,
        )
        request.update({
            "completed_at_utc": utc_now_iso(),
            "http_status": response.status_code,
            "response_sha256": sha256_text(response.text),
        })
        response.raise_for_status()
        raw_results = response.json().get("data", [])
        candidates = [
            {
                "paper_id": paper.get("paperId"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "citation_count": paper.get("citationCount"),
                "url": paper.get("url"),
                "external_ids": paper.get("externalIds", {}),
                "abstract": paper.get("abstract"),
            }
            for paper in raw_results
        ]
        request.update({
            "status": "succeeded",
            "result_count": len(candidates),
            "result_paper_ids": [candidate["paper_id"] for candidate in candidates],
        })
        return candidates, request
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        request.update({
            "status": "request_exception",
            "completed_at_utc": utc_now_iso(),
            "error_type": type(exc).__name__,
            "error_sha256": sha256_text(str(exc)),
        })
        raise ClosureRetrievalError("Semantic Scholar paper search failed", request) from exc


def search_candidate_prior_work(claim: str, limit: int = 20) -> list[dict[str, Any]]:
    """Compatibility wrapper returning candidates without their manifest."""
    candidates, _ = search_candidate_prior_work_with_manifest(claim, limit)
    return candidates


class ClosureRetrievalError(RuntimeError):
    """A retrieval failure whose non-secret request manifest is available."""

    def __init__(self, message: str, request_manifest: dict[str, Any]):
        super().__init__(message)
        self.request_manifest = request_manifest


def citation_neighbors_with_manifest(
    paper_id: str, direction: str, limit: int = 10
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve bounded citation neighborhoods with an auditable API record."""
    if direction not in {"references", "citations"}:
        raise ValueError("direction must be 'references' or 'citations'")
    endpoint = f"{config.SEMANTIC_SCHOLAR_API_BASE_URL}/paper/{paper_id}/{direction}"
    parameters = {"limit": limit, "fields": FIELDS}
    request = _request_manifest(
        endpoint=endpoint, parameters=parameters, started_at_utc=utc_now_iso()
    )
    request.update({"paper_id": paper_id, "direction": direction})
    try:
        _sleep_between_requests()
        response = requests.get(
            endpoint,
            params=parameters,
            headers=_headers(),
            timeout=config.SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS,
        )
        request.update({
            "completed_at_utc": utc_now_iso(),
            "http_status": response.status_code,
            "response_sha256": sha256_text(response.text),
        })
        response.raise_for_status()
        key = "citedPaper" if direction == "references" else "citingPaper"
        neighbors = [
            {
                "paper_id": item.get(key, {}).get("paperId"),
                "title": item.get(key, {}).get("title"),
                "year": item.get(key, {}).get("year"),
                "url": item.get(key, {}).get("url"),
            }
            for item in response.json().get("data", [])
        ]
        request.update({
            "status": "succeeded",
            "result_count": len(neighbors),
            "result_paper_ids": [neighbor["paper_id"] for neighbor in neighbors],
        })
        return neighbors, request
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        request.update({
            "status": "request_exception",
            "completed_at_utc": utc_now_iso(),
            "error_type": type(exc).__name__,
            "error_sha256": sha256_text(str(exc)),
        })
        raise ClosureRetrievalError("Semantic Scholar citation retrieval failed", request) from exc


def citation_neighbors(paper_id: str, direction: str, limit: int = 10) -> list[dict[str, Any]]:
    """Compatibility wrapper returning neighbors without their manifest."""
    neighbors, _ = citation_neighbors_with_manifest(paper_id, direction, limit)
    return neighbors


def _content_tokens(claim: str) -> list[str]:
    tokens = re.findall(r"[^\W\d_][\w-]*", claim, flags=re.UNICODE)
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.casefold()
        if len(token) < 3 or key in _STOP_WORDS or key in seen:
            continue
        seen.add(key)
        unique.append(token)
    return unique


def deterministic_query_variants(claim: str) -> list[dict[str, Any]]:
    """Create stable lexical closure-query variants without an LLM.

    The anchor-pair heuristic is deliberately lexical, not an assertion that
    the selected tokens are a true method/application or concept pair.
    """
    normalized_claim = re.sub(r"\s+", " ", claim).strip()
    tokens = _content_tokens(normalized_claim)
    keyphrase = " ".join(tokens[:8]) or normalized_claim
    anchors = tokens[:2] + tokens[-2:] if len(tokens) > 3 else tokens
    anchor_pair = " ".join(dict.fromkeys(anchors)) or normalized_claim
    base = [
        ("deterministic-verbatim", "verbatim-normalized-claim", normalized_claim),
        ("deterministic-keyphrase", "ordered-content-token-keyphrase", keyphrase),
        ("deterministic-anchor-pair", "first-last-content-token-anchor-pair", anchor_pair),
    ]
    return [
        {
            "query_id": query_id,
            "origin": "deterministic",
            "generator_version": config.CLOSURE_DETERMINISTIC_VARIANT_VERSION,
            "generation_rule": rule,
            "query": query,
            "query_sha256": sha256_text(query),
        }
        for query_id, rule, query in base
        if query
    ]


def generate_llm_query_variants_with_provenance(
    claim: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate optional LLM variants, retaining the exact call provenance."""
    prompt = (
        "Create exactly four short Semantic Scholar search queries for checking whether a research "
        "hypothesis may already be addressed. Cover: keyphrase, concept-pair, method--application, "
        "and synonym-oriented wording. Return only a JSON array of four strings. Hypothesis: " + claim
    )
    generator_manifest: dict[str, Any] = {
        "origin": "llm",
        "prompt_version": config.CLOSURE_QUERY_VARIANT_PROMPT_VERSION,
        "configured_llm": configured_llm_identity(),
        "claim_sha256": sha256_text(claim),
    }
    try:
        result = call_llm_with_provenance(
            prompt,
            temperature=0.0,
            prompt_version=config.CLOSURE_QUERY_VARIANT_PROMPT_VERSION,
            purpose="closure-retrieval-query-variants",
        )
        content = result.text.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1]) if len(lines) >= 2 else content
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise ValueError("Query-variant response is not a JSON array")
        queries = [str(item).strip() for item in parsed if str(item).strip()][:4]
        variants = [
            {
                "query_id": f"llm-{index:02d}",
                "origin": "llm",
                "generator_call_id": result.provenance["call_id"],
                "generator_prompt_version": config.CLOSURE_QUERY_VARIANT_PROMPT_VERSION,
                "query": query,
                "query_sha256": sha256_text(query),
            }
            for index, query in enumerate(queries, start=1)
        ]
        generator_manifest.update({
            "status": "succeeded",
            "llm_call": result.provenance,
            "parsed_variant_count": len(variants),
        })
        return variants, generator_manifest
    except LLMCallError as exc:
        generator_manifest.update({"status": "failed", "llm_call": exc.provenance})
        print(f"[!] Query-variant generation failed: {exc}; continuing with deterministic variants.")
    except Exception as exc:
        generator_manifest.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_sha256": sha256_text(str(exc)),
        })
        print(f"[!] Query-variant generation failed: {exc}; continuing with deterministic variants.")
    return [], generator_manifest


def generate_query_variants(claim: str) -> list[str]:
    """Compatibility wrapper returning optional LLM variants only."""
    variants, _ = generate_llm_query_variants_with_provenance(claim)
    return [variant["query"] for variant in variants]


def _deduplicate_query_variants(
    variants: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve first query occurrence and make duplicate removal inspectable."""
    unique: list[dict[str, Any]] = []
    first_variant_by_key: dict[str, str] = {}
    dropped: list[dict[str, str]] = []
    for variant in variants:
        key = normalized_query_key(variant["query"])
        if not key:
            dropped.append({"query_id": variant["query_id"], "reason": "empty_after_normalization"})
            continue
        if key in first_variant_by_key:
            dropped.append({
                "query_id": variant["query_id"],
                "reason": "normalized_duplicate",
                "deduplicated_into_query_id": first_variant_by_key[key],
            })
            continue
        first_variant_by_key[key] = variant["query_id"]
        unique.append({**variant, "normalized_query_key": key})
    return unique, {
        "normalization": "Unicode case-folding, surrounding whitespace trim, internal whitespace collapse",
        "input_variant_count": len(variants),
        "unique_variant_count": len(unique),
        "dropped_variants": dropped,
    }


def _stable_claim_id(index: int, claim: str, gap: dict[str, Any]) -> str:
    supplied = gap.get("claim_id") or gap.get("candidate_id") or gap.get("id")
    if supplied is not None:
        return str(supplied)
    return f"closure-{index:04d}-{sha256_text(claim)[:12]}"


def build_closure_audit(
    gaps: list[dict[str, Any]],
    limit: int = 20,
    *,
    include_llm_variants: bool = True,
    citation_limit: int = 10,
    citation_candidate_limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve auditable candidate sources; never make a novelty decision."""
    audit: list[dict[str, Any]] = []
    for index, gap in enumerate(gaps):
        claim = str(gap.get("Claim", "")).strip()
        if not claim:
            continue
        claim_id = _stable_claim_id(index, claim, gap)
        deterministic_variants = deterministic_query_variants(claim)
        if include_llm_variants:
            llm_variants, llm_generation = generate_llm_query_variants_with_provenance(claim)
        else:
            llm_variants, llm_generation = [], {
                "origin": "llm",
                "status": "not_requested",
                "reason": "deterministic-only mode",
                "prompt_version": config.CLOSURE_QUERY_VARIANT_PROMPT_VERSION,
                "configured_llm": configured_llm_identity(),
            }
        variants, query_deduplication = _deduplicate_query_variants(
            deterministic_variants + llm_variants
        )
        candidates_by_id: dict[str, dict[str, Any]] = {}
        candidate_first_seen: list[str] = []
        query_executions: list[dict[str, Any]] = []
        retrieval_error = None

        for variant in variants:
            try:
                retrieved, request = search_candidate_prior_work_with_manifest(
                    variant["query"], limit=limit
                )
                query_executions.append({**variant, "api_request": request})
                for rank, candidate in enumerate(retrieved, start=1):
                    paper_id = candidate.get("paper_id")
                    if not paper_id:
                        continue  # Cannot be deduplicated or citation-expanded safely.
                    paper_key = str(paper_id)
                    if paper_key not in candidates_by_id:
                        candidate["first_seen_query_id"] = variant["query_id"]
                        candidate["first_seen_rank"] = rank
                        candidate["retrieved_by_query_ids"] = [variant["query_id"]]
                        candidates_by_id[paper_key] = candidate
                        candidate_first_seen.append(paper_key)
                    else:
                        candidates_by_id[paper_key]["retrieved_by_query_ids"].append(variant["query_id"])
            except ClosureRetrievalError as exc:
                query_executions.append({**variant, "api_request": exc.request_manifest})
                retrieval_error = "one_or_more_query_requests_failed"

        candidates = [candidates_by_id[paper_id] for paper_id in candidate_first_seen]
        citation_requests: list[dict[str, Any]] = []
        for candidate in candidates[:citation_candidate_limit]:
            paper_id = candidate["paper_id"]
            for direction, field in (("references", "backward_references"), ("citations", "forward_citations")):
                try:
                    neighbors, request = citation_neighbors_with_manifest(
                        paper_id, direction, limit=citation_limit
                    )
                    candidate[field] = neighbors
                    citation_requests.append(request)
                except ClosureRetrievalError as exc:
                    candidate.setdefault("citation_retrieval_errors", []).append({
                        "direction": direction,
                        "error_type": type(exc).__name__,
                    })
                    citation_requests.append(exc.request_manifest)

        all_query_succeeded = bool(variants) and all(
            execution["api_request"].get("status") == "succeeded" for execution in query_executions
        )
        status = "retrieved--human-review-required" if all_query_succeeded else "partially-retrieved--human-review-required"
        if not query_executions:
            status = "no-query-variants-generated"
        record = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "candidate_id": index,
            "claim_id": claim_id,
            "claim": claim,
            "claim_sha256": sha256_text(claim),
            "source_signal": gap.get("source"),
            "closure_status": status,
            "candidate_count": len(candidates),
            # Retain this concise backwards-compatible summary beside the full
            # audit manifest below.
            "queries": [
                {"query": execution["query"], "result_count": execution["api_request"].get("result_count")}
                for execution in query_executions
            ],
            "retrieval_error": retrieval_error,
            "candidate_prior_work": candidates,
            "retrieval_manifest": {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "generated_at_utc": utc_now_iso(),
                "deterministic_query_variants": deterministic_variants,
                "llm_query_variant_generation": llm_generation,
                "query_deduplication": query_deduplication,
                "query_executions": query_executions,
                "candidate_deduplication_rule": "Semantic Scholar paperId; keep first occurrence in deterministic/LLM variant order then API rank, retaining all query IDs that retrieved it",
                "candidate_selection_for_citation_expansion": {
                    "limit": citation_candidate_limit,
                    "order": "first-seen unique paper order",
                },
                "citation_neighborhood_policy": {
                    "directions": ["references", "citations"],
                    "per_direction_limit": citation_limit,
                    "requests": citation_requests,
                },
            },
            "review_instruction": (
                "Read candidate sources and their backward/forward citations before judging whether "
                "the claim is addressed, novel, or important. Retrieval rank is not evidence."
            ),
        }
        audit.append(record)
    return audit


def build_closure_run_manifest(
    audit: list[dict[str, Any]],
    *,
    input_path: str | None = None,
    limit: int,
    include_llm_variants: bool,
    citation_limit: int,
    citation_candidate_limit: int,
) -> dict[str, Any]:
    """Create an index over a closure run without duplicating source text."""
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "record_type": "closure_retrieval_run_manifest",
        "generated_at_utc": utc_now_iso(),
        "run_id": config.RUN_ID or None,
        "input": {
            "path": (
                os.path.relpath(input_path, config.DATA_DIR).replace(os.sep, "/")
                if input_path and os.path.exists(input_path) else input_path
            ),
            "sha256": file_sha256(input_path) if input_path and os.path.exists(input_path) else None,
        },
        "retrieval_configuration": {
            "semantic_scholar_api_base_url": config.SEMANTIC_SCHOLAR_API_BASE_URL,
            "paper_search_fields": FIELDS,
            "paper_search_limit": limit,
            "api_timeout_seconds": config.SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS,
            "inter_request_delay_seconds": config.SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS,
            "include_llm_variants": include_llm_variants,
            "deterministic_variant_algorithm_version": config.CLOSURE_DETERMINISTIC_VARIANT_VERSION,
            "llm_variant_prompt_version": config.CLOSURE_QUERY_VARIANT_PROMPT_VERSION,
            "citation_limit": citation_limit,
            "citation_candidate_limit": citation_candidate_limit,
        },
        "claim_count": len(audit),
        "claims": [
            {
                "claim_id": record["claim_id"],
                "claim_sha256": record["claim_sha256"],
                "closure_status": record["closure_status"],
                "candidate_count": record["candidate_count"],
                "query_ids": [
                    execution["query_id"]
                    for execution in record["retrieval_manifest"]["query_executions"]
                ],
            }
            for record in audit
        ],
        "limitations": [
            "The manifest records a retrieval run; Semantic Scholar rankings and coverage can change over time.",
            "LLM query variants are supplementary and cannot be regenerated identically merely from a prompt; their observed outputs and call hashes are retained.",
            "No field in this manifest is a novelty or source-support judgment.",
        ],
    }


def write_closure_outputs(
    audit: list[dict[str, Any]],
    *,
    output_path: str,
    manifest_path: str,
    input_path: str | None,
    limit: int,
    include_llm_variants: bool,
    citation_limit: int,
    citation_candidate_limit: int,
) -> dict[str, Any]:
    """Write audit records plus a separate run-level manifest."""
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    manifest = build_closure_run_manifest(
        audit,
        input_path=input_path,
        limit=limit,
        include_llm_variants=include_llm_variants,
        citation_limit=citation_limit,
        citation_candidate_limit=citation_candidate_limit,
    )
    manifest["output_audit"] = {
        "path": os.path.relpath(output_path, config.DATA_DIR).replace(os.sep, "/"),
        "sha256": file_sha256(output_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve closure-search candidates for KG-TABI hypotheses.")
    parser.add_argument("--input", default=os.path.join(config.GAPS_DIR, "kgtabi_gaps.json"))
    parser.add_argument("--output", default=os.path.join(config.GAPS_DIR, "closure_search_audit.json"))
    parser.add_argument("--manifest", default=os.path.join(config.GAPS_DIR, "closure_search_manifest.json"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--citation-limit", type=int, default=10)
    parser.add_argument("--citation-candidate-limit", type=int, default=5)
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Do not ask an LLM for supplemental query variants.",
    )
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as handle:
        gaps = json.load(handle)
    audit = build_closure_audit(
        gaps,
        args.limit,
        include_llm_variants=not args.deterministic_only,
        citation_limit=args.citation_limit,
        citation_candidate_limit=args.citation_candidate_limit,
    )
    write_closure_outputs(
        audit,
        output_path=args.output,
        manifest_path=args.manifest,
        input_path=args.input,
        limit=args.limit,
        include_llm_variants=not args.deterministic_only,
        citation_limit=args.citation_limit,
        citation_candidate_limit=args.citation_candidate_limit,
    )
    print(f"[+] Wrote {len(audit)} closure-search records to {args.output}")
    print(f"[+] Wrote closure-search manifest to {args.manifest}")


if __name__ == "__main__":
    main()
