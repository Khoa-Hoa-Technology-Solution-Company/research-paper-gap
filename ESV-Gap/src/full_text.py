"""Bounded open-access full-text enrichment for top evidence-cell sources."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable

import requests

from src.utils import get_logger, load_json, load_jsonl, save_json, save_jsonl


logger = get_logger("full_text")


def _paper_id(paper: dict[str, Any]) -> str:
    return str(paper.get("paperId") or paper.get("paper_id") or paper.get("id") or "")


def _full_text_present(paper: dict[str, Any], minimum: int) -> bool:
    return any(
        len(str(paper.get(field) or "").strip()) >= minimum
        for field in ("full_text", "fulltext", "text", "content")
    )


def _open_access_pdf_url(paper: dict[str, Any]) -> str:
    record = paper.get("openAccessPdf") or paper.get("open_access_pdf") or {}
    if not isinstance(record, dict):
        return ""
    url = str(record.get("url") or "").strip()
    status = str(record.get("status") or "").upper()
    # Semantic Scholar can return a DOI landing page in this field. Only fetch
    # URLs advertised as open access; the response is still verified as PDF.
    return url if url and status not in {"CLOSED", "PAYWALLED"} else ""


def extract_pdf_text(payload: bytes, max_pages: int = 80) -> str:
    """Extract text from a verified PDF byte stream with a bounded page count."""
    if not payload.startswith(b"%PDF"):
        return ""
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF is unavailable; skipping full-text extraction.")
        return ""
    try:
        document = fitz.open(stream=payload, filetype="pdf")
        pages = [
            document.load_page(index).get_text("text")
            for index in range(min(document.page_count, max_pages))
        ]
        document.close()
        return "\n".join(pages).strip()
    except Exception as exc:
        logger.warning("Could not parse an open-access PDF: %s", exc)
        return ""


def enrich_candidate_source_full_text(
    config: dict[str, Any],
    request_get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    """Fetch direct OA PDFs only for sources of the strongest candidates."""
    settings = config.get("full_text", {})
    if not settings.get("enabled", True):
        return {"performed": False, "reason": "disabled", "enriched": 0}
    corpus_path = (
        Path(config["paths"]["processed_data"]) / "corpus_filtered.jsonl"
    )
    gaps_path = Path(config["paths"]["outputs"]) / "detected_gaps_raw.json"
    if not corpus_path.exists() or not gaps_path.exists():
        return {"performed": False, "reason": "inputs_missing", "enriched": 0}

    papers = list(load_jsonl(corpus_path))
    paper_index = {_paper_id(paper): paper for paper in papers if _paper_id(paper)}
    raw_gaps = load_json(gaps_path)
    gaps = list(raw_gaps.get("evidence_gaps", [])) + list(
        raw_gaps.get("missing_links", [])
    )
    max_candidates = max(int(settings.get("max_candidates", 10)), 0)
    max_papers = max(int(settings.get("max_source_papers", 20)), 0)
    minimum = int(
        config.get("gap_certification", {}).get("min_full_text_characters", 1000)
    )
    target_ids = []
    seen = set()
    certifiable_candidates = [
        candidate for candidate in gaps
        if candidate.get("candidate_quality", {})
        .get("certificate_readiness", {})
        .get("independent_sources_ready", False)
    ]
    for candidate in sorted(
        certifiable_candidates,
        key=lambda item: item.get("candidate_quality", {}).get("score", 0.0),
        reverse=True,
    )[:max_candidates]:
        for paper_id in candidate.get("supporting_paper_ids", []):
            paper_id = str(paper_id)
            if paper_id and paper_id not in seen:
                target_ids.append(paper_id)
                seen.add(paper_id)
            if len(target_ids) >= max_papers:
                break
        if len(target_ids) >= max_papers:
            break

    enriched = 0
    attempted = 0
    skipped_no_direct_pdf = 0
    errors = []
    max_bytes = int(settings.get("max_pdf_bytes", 20_000_000))
    timeout = float(settings.get("request_timeout_seconds", 20))
    max_pages = int(settings.get("max_pages", 80))
    for paper_id in target_ids:
        paper = paper_index.get(paper_id, {})
        if _full_text_present(paper, minimum):
            continue
        url = _open_access_pdf_url(paper)
        if not url:
            skipped_no_direct_pdf += 1
            continue
        attempted += 1
        try:
            response = request_get(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers={"User-Agent": "ESV-Gap/1.0 research evidence retrieval"},
            )
            response.raise_for_status()
            payload = bytes(response.content)
            if len(payload) > max_bytes:
                errors.append(f"{paper_id}:pdf_too_large")
                continue
            text = extract_pdf_text(payload, max_pages=max_pages)
            if len(text) < minimum:
                errors.append(f"{paper_id}:insufficient_extractable_text")
                continue
            paper["full_text"] = text
            paper["full_text_provenance"] = {
                "source_url": url,
                "retrieval_kind": "direct_open_access_pdf",
                "characters": len(text),
            }
            enriched += 1
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append(f"{paper_id}:{exc}")

    if enriched:
        save_jsonl(papers, corpus_path)
    report = {
        "schema_version": 1,
        "performed": True,
        "target_paper_count": len(target_ids),
        "attempted_download_count": attempted,
        "skipped_without_direct_oa_pdf": skipped_no_direct_pdf,
        "enriched": enriched,
        "errors": errors,
    }
    save_json(
        report,
        Path(config["paths"]["outputs"]) / "full_text_enrichment.json",
    )
    logger.info("Full-text enrichment: %s", report)
    return report
