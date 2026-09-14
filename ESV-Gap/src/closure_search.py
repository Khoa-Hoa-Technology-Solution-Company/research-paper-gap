"""Candidate-specific scholarly searches for potentially closing literature."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from src.collect import PAPER_FIELDS, SEARCH_URL
from src.entity_normalization import canonical_entity_label
from src.utils import get_logger


logger = get_logger("closure_search")
OPENALEX_WORKS_URL = "https://api.openalex.org/works"


class ClosureSearchError(RuntimeError):
    """Raised when a candidate-level closure search cannot be completed."""


def _compact_search_label(value: Any, max_words: int = 10) -> str:
    """Keep scholarly API queries focused when an extracted entity is a clause."""
    label = canonical_entity_label(value)
    label = re.split(r"[;,]", label, maxsplit=1)[0].strip()
    words = label.split()
    return " ".join(words[:max_words])


def candidate_search_query(candidate: dict[str, Any], domain: str = "") -> str:
    """Build a deterministic query from the concepts defining a candidate."""
    gap_type = candidate.get("type")
    if gap_type == "evidence_gap":
        scope_anchor = (
            domain if candidate.get("semantic_scope") == "field"
            else candidate.get("subject")
        )
        entities = [scope_anchor, candidate.get("missing_capability")]
    elif gap_type == "missing_link":
        entities = [candidate.get("head"), candidate.get("tail")]
    elif gap_type == "temporal_decay":
        entities = [candidate.get("concept")]
    else:
        entities = candidate.get("key_concepts", [])[:3]

    terms = []
    seen = set()
    for entity in entities:
        label = _compact_search_label(entity)
        key = label.casefold()
        if label and key not in seen:
            terms.append(label)
            seen.add(key)
    if len(terms) == 1 and domain and domain.casefold() not in seen:
        terms.append(_compact_search_label(domain))
    return " ".join(terms).strip()


def candidate_search_queries(
    candidate: dict[str, Any],
    domain: str = "",
    max_queries: int = 3,
) -> list[str]:
    """Build complementary exact, problem-focused, and domain queries."""
    primary = candidate_search_query(candidate, domain)
    if domain and canonical_entity_label(domain).casefold() not in primary.casefold():
        primary = " ".join((primary, _compact_search_label(domain))).strip()
    gap_type = candidate.get("type")
    if gap_type == "evidence_gap":
        subject = _compact_search_label(candidate.get("subject"))
        missing = _compact_search_label(candidate.get("missing_capability"))
        scope_anchor = domain if candidate.get("semantic_scope") == "field" else subject
        variants = [
            primary,
            " ".join(value for value in (missing, domain, "solution") if value),
            " ".join(value for value in (scope_anchor, missing, domain, "improvement") if value),
        ]
    elif gap_type == "missing_link":
        head = _compact_search_label(candidate.get("head"))
        tail = _compact_search_label(candidate.get("tail"))
        variants = [
            primary,
            " ".join(value for value in (head, tail, domain) if value),
            " ".join(value for value in (head, tail, domain, "benchmark") if value),
        ]
    else:
        variants = [primary]
    queries = []
    seen = set()
    for query in variants:
        normalized = " ".join(str(query).split()).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            queries.append(normalized)
            seen.add(key)
    return queries[:max(1, int(max_queries))]


def search_semantic_scholar(
    query: str,
    year_range: list[int] | tuple[int, int],
    api_key: str = "",
    max_results: int = 20,
    request_timeout: float = 15,
    max_retries: int = 2,
    max_retry_wait: float = 15,
    request_get: Callable[..., Any] = requests.get,
) -> list[dict[str, Any]]:
    """Run a bounded relevance-ranked Semantic Scholar paper search."""
    if not query:
        return []
    headers = {"x-api-key": api_key} if api_key else {}
    params = {
        "query": query,
        "fields": PAPER_FIELDS,
        "year": f"{int(year_range[0])}-{int(year_range[1])}",
        "limit": min(max(int(max_results), 1), 100),
    }
    retries = max(1, int(max_retries))
    for attempt in range(retries):
        try:
            response = request_get(
                SEARCH_URL,
                params=params,
                headers=headers,
                timeout=request_timeout,
            )
            if response.status_code == 429:
                if attempt + 1 >= retries:
                    raise ClosureSearchError(
                        f"Semantic Scholar remained rate-limited for query '{query}'."
                    )
                retry_after = float(response.headers.get("Retry-After", 5))
                wait = min(max(retry_after, 0.0), float(max_retry_wait))
                logger.warning(
                    "Closure search rate limited; retry %d/%d in %.1fs",
                    attempt + 1,
                    retries,
                    wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            payload = response.json()
            return list(payload.get("data", []))[:max_results]
        except ClosureSearchError:
            raise
        except (requests.RequestException, ValueError, TypeError) as exc:
            if attempt + 1 >= retries:
                raise ClosureSearchError(
                    f"Semantic Scholar closure search failed for '{query}': {exc}"
                ) from exc
            wait = min(2 ** attempt, float(max_retry_wait))
            logger.warning(
                "Closure search failed; retry %d/%d in %.1fs: %s",
                attempt + 1,
                retries,
                wait,
                exc,
            )
            time.sleep(wait)
    return []


def _openalex_abstract(inverted_index: Any) -> str:
    """Reconstruct an OpenAlex abstract from its positional inverted index."""
    if not isinstance(inverted_index, dict):
        return ""
    positioned = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            try:
                positioned.append((int(position), str(word)))
            except (TypeError, ValueError):
                continue
    positioned.sort()
    return " ".join(word for _, word in positioned)


def search_openalex(
    query: str,
    year_range: list[int] | tuple[int, int],
    api_key: str = "",
    max_results: int = 20,
    request_timeout: float = 15,
    max_retries: int = 2,
    max_retry_wait: float = 15,
    request_get: Callable[..., Any] = requests.get,
) -> list[dict[str, Any]]:
    """Run a bounded OpenAlex works search and normalize paper metadata."""
    if not query:
        return []
    params = {
        "search": query,
        "filter": (
            f"from_publication_date:{int(year_range[0])}-01-01,"
            f"to_publication_date:{int(year_range[1])}-12-31"
        ),
        "per-page": min(max(int(max_results), 1), 100),
        "select": (
            "id,doi,title,display_name,publication_year,"
            "abstract_inverted_index,cited_by_count,primary_location"
        ),
    }
    if api_key:
        params["api_key"] = api_key
    retries = max(1, int(max_retries))
    for attempt in range(retries):
        try:
            response = request_get(
                OPENALEX_WORKS_URL,
                params=params,
                timeout=request_timeout,
            )
            if response.status_code == 429:
                if attempt + 1 >= retries:
                    raise ClosureSearchError(
                        f"OpenAlex remained rate-limited for query '{query}'."
                    )
                retry_after = float(response.headers.get("Retry-After", 5))
                wait = min(max(retry_after, 0.0), float(max_retry_wait))
                logger.warning(
                    "OpenAlex closure search rate limited; retry %d/%d in %.1fs",
                    attempt + 1,
                    retries,
                    wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            results = list(response.json().get("results", []))[:max_results]
            papers = []
            for work in results:
                location = work.get("primary_location") or {}
                papers.append({
                    "paperId": str(work.get("id") or work.get("doi") or ""),
                    "title": work.get("title") or work.get("display_name") or "",
                    "abstract": _openalex_abstract(
                        work.get("abstract_inverted_index")
                    ),
                    "year": work.get("publication_year"),
                    "citationCount": work.get("cited_by_count", 0),
                    "url": location.get("landing_page_url") or work.get("doi") or "",
                    "externalIds": {
                        "OpenAlex": work.get("id"),
                        "DOI": work.get("doi"),
                    },
                })
            return papers
        except ClosureSearchError:
            raise
        except (requests.RequestException, ValueError, TypeError, RuntimeError) as exc:
            if attempt + 1 >= retries:
                raise ClosureSearchError(
                    f"OpenAlex closure search failed for '{query}': {exc}"
                ) from exc
            wait = min(2 ** attempt, float(max_retry_wait))
            logger.warning(
                "OpenAlex closure search failed; retry %d/%d in %.1fs: %s",
                attempt + 1,
                retries,
                wait,
                exc,
            )
            time.sleep(wait)
    return []


def search_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Search one candidate and return an auditable completion record."""
    validation = config.get("gap_validation", {})
    settings = validation.get("external_closure", {})
    domain = config.get("project", {}).get("domain", "")
    queries = candidate_search_queries(
        candidate,
        domain,
        max_queries=int(settings.get("queries_per_candidate", 3)),
    )
    query = queries[0] if queries else ""
    year_range = config.get("collection", {}).get("year_range", [1900, 2100])
    audit_context = {
        "source": "Semantic Scholar Academic Graph API with OpenAlex fallback",
        "year_range": list(year_range),
        "searched_at": datetime.now(timezone.utc).isoformat(),
    }
    if not settings.get("enabled", False):
        return {
            **audit_context,
            "performed": False,
            "query": query,
            "papers": [],
            "error": "disabled",
        }
    try:
        papers_by_id = {}
        query_records = []
        completed = 0
        errors = []
        for position, candidate_query in enumerate(queries):
            try:
                provider = "Semantic Scholar Academic Graph API"
                fallback_reason = None
                semantic_scholar_key = config.get("api_keys", {}).get(
                    "semantic_scholar", ""
                )
                prefer_openalex = bool(
                    settings.get("prefer_openalex_without_semantic_scholar_key", True)
                    and not semantic_scholar_key
                )
                if prefer_openalex:
                    provider = "OpenAlex API"
                    fallback_reason = "semantic_scholar_api_key_not_configured"
                    query_papers = search_openalex(
                        query=candidate_query,
                        year_range=year_range,
                        api_key=config.get("api_keys", {}).get("openalex", ""),
                        max_results=int(settings.get("max_results_per_query", 20)),
                        request_timeout=float(settings.get("request_timeout_seconds", 15)),
                        max_retries=int(settings.get("max_retries", 2)),
                        max_retry_wait=float(settings.get("max_retry_wait_seconds", 15)),
                    )
                else:
                    try:
                        query_papers = search_semantic_scholar(
                            query=candidate_query,
                            year_range=year_range,
                            api_key=semantic_scholar_key,
                            max_results=int(settings.get("max_results_per_query", 20)),
                            request_timeout=float(settings.get("request_timeout_seconds", 15)),
                            max_retries=int(settings.get("max_retries", 2)),
                            max_retry_wait=float(settings.get("max_retry_wait_seconds", 15)),
                        )
                    except ClosureSearchError as semantic_scholar_error:
                        if not settings.get("openalex_fallback", True):
                            raise
                        fallback_reason = str(semantic_scholar_error)
                        provider = "OpenAlex API"
                        logger.warning(
                            "Semantic Scholar unavailable; using OpenAlex for '%s'.",
                            candidate_query,
                        )
                        query_papers = search_openalex(
                            query=candidate_query,
                            year_range=year_range,
                            api_key=config.get("api_keys", {}).get("openalex", ""),
                            max_results=int(settings.get("max_results_per_query", 20)),
                            request_timeout=float(settings.get("request_timeout_seconds", 15)),
                            max_retries=int(settings.get("max_retries", 2)),
                            max_retry_wait=float(settings.get("max_retry_wait_seconds", 15)),
                        )
                completed += 1
                query_records.append({
                    "query": candidate_query,
                    "performed": True,
                    "provider": provider,
                    "fallback_reason": fallback_reason,
                    "retrieved_count": len(query_papers),
                    "error": None,
                })
                for paper in query_papers:
                    paper_id = str(
                        paper.get("paperId")
                        or paper.get("paper_id")
                        or paper.get("id")
                        or f"query-{position}-{len(papers_by_id)}"
                    )
                    papers_by_id.setdefault(paper_id, paper)
            except ClosureSearchError as exc:
                errors.append(str(exc))
                query_records.append({
                    "query": candidate_query,
                    "performed": False,
                    "provider": None,
                    "retrieved_count": 0,
                    "error": str(exc),
                })
            if position < len(queries) - 1:
                time.sleep(max(float(settings.get("delay_between_queries_seconds", 2.0)), 0.0))
        papers = list(papers_by_id.values())
        return {
            **audit_context,
            "performed": completed == len(queries) and bool(queries),
            "query": query,
            "queries": query_records,
            "completed_query_count": completed,
            "required_query_count": len(queries),
            "retrieved_count": len(papers),
            "papers": papers,
            "error": "; ".join(errors) if errors else None,
        }
    except ClosureSearchError as exc:
        logger.warning("Candidate closure search unavailable: %s", exc)
        return {
            **audit_context,
            "performed": False,
            "query": query,
            "retrieved_count": 0,
            "papers": [],
            "error": str(exc),
        }
