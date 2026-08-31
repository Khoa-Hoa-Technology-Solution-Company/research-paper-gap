"""Evidence certificates for corpus-bounded research-gap conclusions.

The certificate is intentionally stronger than a candidate score.  It does not
claim that no relevant study exists worldwide; it certifies that a precise
PMCOST-style evidence cell is inadequately supported in a documented corpus
after predefined counterevidence searches.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


CERTIFICATE_SCHEMA_VERSION = 1
_TEXT_FIELDS = ("full_text", "fulltext", "text", "content")


def document_id(document: dict[str, Any]) -> str:
    return str(
        document.get("paperId") or document.get("paper_id") or document.get("id") or ""
    )


def source_text_level(document: dict[str, Any], min_full_text_characters: int) -> str:
    """Return the best verifiable evidence level available for one source."""
    for field in _TEXT_FIELDS:
        if len(str(document.get(field) or "").strip()) >= min_full_text_characters:
            return "full_text"
    if str(document.get("abstract") or "").strip():
        return "abstract"
    return "metadata_only"


def _fingerprint(documents: list[dict[str, Any]]) -> str:
    identities = sorted(document_id(document) for document in documents if document_id(document))
    return hashlib.sha256("|".join(identities).encode("utf-8")).hexdigest()


def build_gap_certificate(
    synthesis_result: dict[str, Any],
    documents: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic, inspectable certificate for one synthesis result."""
    settings = config.get("gap_certification", {})
    enabled = bool(settings.get("enabled", False))
    validation = synthesis_result.get("source_candidate", {}).get("validation", {}) or {}
    decision = synthesis_result.get("decision", {}) or {}
    source_ids = [str(value) for value in synthesis_result.get("supporting_paper_ids", []) if value]
    if not source_ids:
        source_ids = [
            str(value) for value in validation.get("supporting_paper_ids", []) if value
        ]
    source_ids = sorted(set(source_ids))
    paper_index = {document_id(document): document for document in documents}
    source_records = []
    min_chars = int(settings.get("min_full_text_characters", 1000))
    for paper_id in source_ids:
        document = paper_index.get(paper_id, {})
        source_records.append({
            "paper_id": paper_id,
            "title": document.get("title", ""),
            "year": document.get("year") or document.get("publication_year"),
            "text_level": source_text_level(document, min_chars) if document else "missing",
        })

    source_level_required = str(settings.get("min_source_text_level", "full_text"))
    require_full_text = source_level_required == "full_text"
    source_text_covered = bool(source_records) and all(
        record["text_level"] == "full_text" if require_full_text
        else record["text_level"] in {"full_text", "abstract"}
        for record in source_records
    )

    closure = synthesis_result.get("external_closure", {}) or {}
    closure_record = closure.get("record", {}) or {}
    # Current closure-search records use ``queries``; accept
    # ``query_records`` as well so old frozen runs remain auditable.  A legacy
    # record without per-query evidence deliberately cannot pass this gate.
    query_records = (
        closure_record.get("query_records")
        or closure_record.get("queries")
        or []
    )
    required_queries = int(settings.get(
        "min_external_queries",
        config.get("gap_synthesis", {}).get("min_external_queries", 2),
    ))
    completed_queries = int(closure.get("completed_query_count", 0))
    all_queries_completed = bool(query_records) and all(
        item.get("performed") is True for item in query_records
    ) and len(query_records) >= required_queries
    if not settings.get("require_all_queries_completed", True):
        all_queries_completed = True

    source_resolution = list(
        synthesis_result.get("source_resolution_edges", []) or []
    )
    source_resolution.extend(
        synthesis_result.get("source_resolution_documents", []) or []
    )
    closure_hits = list(closure.get("closure_hits", []) or [])
    local_hits = list(validation.get("closure_hits", []) or [])
    evidence_sources = len(source_ids)
    gates = {
        "synthesis_contract_passed": bool(decision.get("passed")),
        "independent_limitation_sources": evidence_sources >= int(
            settings.get("min_independent_limitation_sources", 2)
        ),
        "source_text_evidence_sufficient": source_text_covered,
        "external_counterevidence_search_complete": (
            bool(closure.get("performed"))
            and completed_queries >= required_queries
            and all_queries_completed
        ),
        "counterevidence_free": not source_resolution and not closure_hits and not local_hits,
        "answerable_pmcost": bool(synthesis_result.get("answerable")),
        "frozen_corpus_fingerprint": bool(_fingerprint(documents)),
    }
    failed_gates = [name for name, passed in gates.items() if not passed]
    identity = {
        "gap_id": synthesis_result.get("gap_id", ""),
        "corpus_fingerprint": _fingerprint(documents),
        "snapshot_date": config.get("gap_validation", {}).get("snapshot_date", ""),
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
    }
    certificate_id = "gc_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    passed = enabled and not failed_gates
    return {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "certificate_id": certificate_id,
        "enabled": enabled,
        "verdict": (
            "certified_corpus_bounded_research_gap"
            if passed else "not_certified_as_corpus_bounded_research_gap"
        ),
        "passed": passed,
        "scope": {
            "claim_scope": "bounded screened corpus and documented closure search",
            "snapshot_date": identity["snapshot_date"],
            "corpus_fingerprint_sha256": identity["corpus_fingerprint"],
            "screened_paper_count": len(documents),
        },
        "gates": gates,
        "failed_gates": failed_gates,
        "source_evidence": source_records,
        "external_search": {
            "performed": bool(closure.get("performed")),
            "completed_query_count": completed_queries,
            "required_query_count": required_queries,
            "all_query_records_completed": all_queries_completed,
            "retrieved_paper_ids": closure_record.get("retrieved_paper_ids", []),
            "closure_hit_count": len(closure_hits),
        },
        "counterevidence": {
            "source_resolution_count": len(source_resolution),
            "local_closure_hit_count": len(local_hits),
            "external_closure_hit_count": len(closure_hits),
        },
        "limitations": [
            "This certificate does not prove global non-existence of research.",
            "It certifies a corpus-bounded evidence gap only under the recorded search, corpus, and text-coverage conditions.",
        ],
    }
