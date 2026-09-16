"""External absence verification for research-gap candidates.

The local coverage screen in :mod:`src.validate_gaps` asks whether the
*screened corpus* already covers a candidate.  For an absence claim that test
is circular: the same corpus produced the knowledge graph, so a relation that
is missing from the graph is usually missing from the corpus as well.  Passing
it is therefore weak evidence that nobody has studied the relation.

This module breaks the circularity by probing scholarly indexes that were never
used to build the graph.  A candidate can only be *weakened* by an external
hit, never strengthened by a silent failure: probe errors, unreachable indexes,
and unconfigured sources all fail closed to ``verification_failed``.

Verdicts
--------
``refuted_by_external_literature``
    An indexed paper co-mentions the candidate's entities, so the claimed
    absence does not hold outside the screened corpus.
``absence_corroborated``
    Every configured probe ran and no co-mentioning paper was found.
``verification_failed``
    At least one probe could not be completed; absence remains unproven.
``not_applicable``
    The candidate type has no defined external probe.

Every probe records its exact query string, source, timestamp, raw index count,
and post-filtered hit count, so a reviewer can re-run the search by hand.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from dotenv import find_dotenv, load_dotenv

from src.entity_normalization import (
    canonical_entity_key,
    canonical_entity_label,
    entity_tokens,
    mentions_entity,
)
from src.utils import ensure_dir, get_logger, load_json, save_json

load_dotenv(find_dotenv())
logger = get_logger("external_verification")

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

REFUTED = "refuted_by_external_literature"
CORROBORATED = "absence_corroborated"
FAILED = "verification_failed"
NOT_APPLICABLE = "not_applicable"


class ExternalProbeError(RuntimeError):
    """Raised when an index cannot be queried after bounded retries."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _decode_openalex_abstract(inverted: dict[str, list[int]] | None) -> str:
    """Rebuild an abstract from OpenAlex's inverted index."""
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, offsets in inverted.items():
        for offset in offsets:
            positions[offset] = word
    return " ".join(positions[key] for key in sorted(positions))


def _http_json(url: str, headers: dict[str, str], timeout: float, retries: int, backoff: float) -> dict[str, Any]:
    """GET JSON with bounded retries.  Raises rather than returning partial data."""
    import requests

    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code in (429, 503):
                retry_after = 5.0
                try:
                    retry_after = float(response.headers.get("Retry-After", 5.0))
                except (ValueError, TypeError):
                    pass
                wait_time = max(retry_after, backoff * (2 ** (attempt - 1)), 5.0)
                time.sleep(min(wait_time, 20.0))
                last_error = ExternalProbeError(f"index throttled with HTTP {response.status_code}")
                continue
            response.raise_for_status()
            return response.json()
        except Exception as error:  # network, decode, and throttling all fail closed
            last_error = error
            if attempt >= attempts:
                break
            time.sleep(min(backoff * (2 ** (attempt - 1)), 20.0))
    raise ExternalProbeError(f"{url} failed after {attempts} attempts: {last_error}")


class ProbeCache:
    """Disk-backed cache so re-runs are free and byte-identical.

    Keys are SHA-256 digests of the source and query, which keeps the audit
    trail stable across runs even when candidate ordering changes.
    """

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._entries: dict[str, Any] = {}
        if self.path and self.path.exists():
            try:
                loaded = load_json(self.path)
                if isinstance(loaded, dict):
                    self._entries = loaded
            except (json.JSONDecodeError, OSError):
                logger.warning("Ignoring unreadable probe cache at %s", self.path)

    @staticmethod
    def key(source: str, query: str) -> str:
        digest = hashlib.sha256(f"{source}|{query}".encode("utf-8")).hexdigest()
        return digest[:32]

    def get(self, source: str, query: str) -> dict[str, Any] | None:
        return self._entries.get(self.key(source, query))

    def put(self, source: str, query: str, value: dict[str, Any]) -> None:
        self._entries[self.key(source, query)] = value

    def flush(self) -> None:
        if self.path:
            ensure_dir(self.path.parent)
            save_json(self._entries, self.path)


def openalex_searcher(settings: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """Build an OpenAlex searcher.  The mailto contact is OpenAlex etiquette."""
    contact = str(settings.get("openalex_contact_email", "") or "")
    timeout = float(settings.get("request_timeout_seconds", 25))
    retries = int(settings.get("max_retries", 3))
    backoff = float(settings.get("retry_backoff_seconds", 2.0))
    per_page = int(settings.get("max_records_per_probe", 5))
    headers = {"User-Agent": f"ESV-Gap/1.0 (+{contact})" if contact else "ESV-Gap/1.0"}

    def search(query: str, mode: str = "search") -> dict[str, Any]:
        params: dict[str, str] = {"filter": f"title_and_abstract.search:{query}"}
        if contact:
            params["mailto"] = contact
        if mode == "year_counts":
            params["group_by"] = "publication_year"
        else:
            params["per-page"] = str(max(1, per_page))
        url = f"{OPENALEX_WORKS_URL}?{urllib.parse.urlencode(params)}"
        payload = _http_json(url, headers, timeout, retries, backoff)

        if mode == "year_counts":
            counts: dict[int, int] = {}
            for bucket in payload.get("group_by", []) or []:
                try:
                    counts[int(bucket.get("key"))] = int(bucket.get("count", 0))
                except (TypeError, ValueError):
                    continue
            return {"raw_count": sum(counts.values()), "records": [], "year_counts": counts}

        records = []
        for work in payload.get("results", []) or []:
            records.append({
                "id": str(work.get("id") or ""),
                "title": str(work.get("title") or ""),
                "abstract": _decode_openalex_abstract(work.get("abstract_inverted_index")),
                "year": work.get("publication_year"),
                "doi": str(work.get("doi") or ""),
            })
        return {
            "raw_count": int((payload.get("meta") or {}).get("count", 0)),
            "records": records,
        }

    return search


def semantic_scholar_searcher(settings: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """Build a Semantic Scholar searcher.  Year aggregation is unsupported."""
    timeout = float(settings.get("request_timeout_seconds", 25))
    retries = int(settings.get("max_retries", 3))
    backoff = float(settings.get("retry_backoff_seconds", 2.0))
    per_page = int(settings.get("max_records_per_probe", 5))
    api_key = str(settings.get("semantic_scholar_api_key", "") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "") or "")
    headers = {"x-api-key": api_key} if api_key else {}

    def search(query: str, mode: str = "search") -> dict[str, Any]:
        if mode == "year_counts":
            raise ExternalProbeError("Semantic Scholar does not aggregate by year")
        params = {
            "query": query,
            "fields": "paperId,title,abstract,year,externalIds",
            "limit": str(max(1, per_page)),
        }
        url = f"{SEMANTIC_SCHOLAR_SEARCH_URL}?{urllib.parse.urlencode(params)}"
        payload = _http_json(url, headers, timeout, retries, backoff)
        records = []
        for paper in payload.get("data", []) or []:
            records.append({
                "id": str(paper.get("paperId") or ""),
                "title": str(paper.get("title") or ""),
                "abstract": str(paper.get("abstract") or ""),
                "year": paper.get("year"),
                "doi": str((paper.get("externalIds") or {}).get("DOI") or ""),
            })
        return {"raw_count": int(payload.get("total", 0)), "records": records}

    return search


def entity_phrases(candidate: dict[str, Any], entity: str, max_aliases: int = 2) -> list[str]:
    """Return the canonical label plus recorded aliases for one entity."""
    label = canonical_entity_label(entity)
    phrases = [label] if label else []
    aliases = candidate.get("entity_aliases", {}) or {}
    for alias_entity, values in aliases.items():
        if canonical_entity_key(alias_entity) != canonical_entity_key(entity):
            continue
        for value in values:
            alias = canonical_entity_label(value)
            if alias and canonical_entity_key(alias) != canonical_entity_key(label):
                phrases.append(alias)
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        key = canonical_entity_key(phrase)
        if key not in seen:
            unique.append(phrase)
            seen.add(key)
    return unique[: 1 + max(0, max_aliases)]


def pair_queries(left_phrases: Iterable[str], right_phrases: Iterable[str]) -> list[str]:
    """Build deterministic quoted AND queries for every phrase combination."""
    queries = []
    for left in left_phrases:
        for right in right_phrases:
            queries.append(f'("{left}" AND "{right}")')
    return queries


def single_queries(phrases: Iterable[str]) -> list[str]:
    return [f'"{phrase}"' for phrase in phrases]


def confirm_co_mentions(
    records: Iterable[dict[str, Any]],
    entity_groups: list[list[str]],
    coverage: float,
) -> list[dict[str, Any]]:
    """Keep only records that genuinely mention every entity group.

    Index boolean semantics vary between providers and cannot be trusted, so a
    raw hit count is never used as a verdict on its own.  This applies the same
    lexical test as the local coverage screen to the returned records.
    """
    confirmed = []
    for record in records:
        text = " ".join(
            str(record.get(key, "") or "") for key in ("title", "abstract")
        )
        text_token_set = entity_tokens(text)
        if not text_token_set:
            continue
        matched_all = True
        for group in entity_groups:
            token_sets = [entity_tokens(phrase) for phrase in group]
            if not any(
                mentions_entity(text_token_set, tokens, coverage) for tokens in token_sets if tokens
            ):
                matched_all = False
                break
        if matched_all:
            confirmed.append({
                "id": record.get("id", ""),
                "title": record.get("title", ""),
                "year": record.get("year"),
                "doi": record.get("doi", ""),
            })
    return confirmed


def run_probe(
    source: str,
    search: Callable[..., dict[str, Any]],
    query: str,
    entity_groups: list[list[str]],
    coverage: float,
    cache: ProbeCache,
    mode: str = "search",
) -> dict[str, Any]:
    """Execute one recorded probe, returning an auditable record."""
    cached = cache.get(source, query)
    if cached is not None:
        return {**cached, "cached": True}

    probe: dict[str, Any] = {
        "source": source,
        "query": query,
        "mode": mode,
        "retrieved_at": _utc_now(),
    }
    try:
        payload = search(query, mode)
    except Exception as error:
        probe.update({"ok": False, "error": str(error)[:300]})
        return probe

    probe["ok"] = True
    probe["raw_count"] = int(payload.get("raw_count", 0))
    if mode == "year_counts":
        probe["year_counts"] = {str(k): v for k, v in (payload.get("year_counts") or {}).items()}
    else:
        confirmed = confirm_co_mentions(payload.get("records", []), entity_groups, coverage)
        probe["inspected_records"] = len(payload.get("records", []))
        probe["confirmed_hits"] = confirmed
        probe["confirmed_hit_count"] = len(confirmed)
    cache.put(source, query, probe)
    return {**probe, "cached": False}


def build_searchers(settings: dict[str, Any]) -> dict[str, Callable[..., dict[str, Any]]]:
    """Instantiate the searchers named in configuration, in listed order."""
    factories = {
        "openalex": openalex_searcher,
        "semantic_scholar": semantic_scholar_searcher,
    }
    searchers: dict[str, Callable[..., dict[str, Any]]] = {}
    for name in settings.get("sources", []) or []:
        factory = factories.get(str(name))
        if factory is None:
            logger.warning("Unknown external verification source '%s'; ignoring", name)
            continue
        searchers[str(name)] = factory(settings)
    return searchers


def _verdict_from_probes(probes: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Fail closed: any refuting hit refutes, any failed probe blocks corroboration."""
    refuting = [
        hit
        for probe in probes
        for hit in probe.get("confirmed_hits", [])
    ]
    if refuting:
        return REFUTED, refuting
    if not probes or any(not probe.get("ok") for probe in probes):
        return FAILED, []
    return CORROBORATED, []


def _verify_missing_link(
    candidate: dict[str, Any],
    searchers: dict[str, Callable[..., dict[str, Any]]],
    settings: dict[str, Any],
    cache: ProbeCache,
) -> dict[str, Any]:
    head_phrases = entity_phrases(candidate, candidate.get("head", ""))
    tail_phrases = entity_phrases(candidate, candidate.get("tail", ""))
    if not head_phrases or not tail_phrases:
        return {"verdict": FAILED, "probes": [], "reason": "missing endpoint labels"}

    coverage = float(settings.get("co_mention_token_coverage", 0.60))
    max_queries = int(settings.get("max_queries_per_candidate", 4))
    entity_groups = [head_phrases, tail_phrases]
    queries = pair_queries(head_phrases, tail_phrases)[:max_queries]

    probes = []
    for source, search in searchers.items():
        for query in queries:
            probes.append(
                run_probe(source, search, query, entity_groups, coverage, cache)
            )
    verdict, refuting = _verdict_from_probes(probes)
    return {"verdict": verdict, "probes": probes, "refuting_papers": refuting}


def _verify_temporal_decay(
    candidate: dict[str, Any],
    searchers: dict[str, Callable[..., dict[str, Any]]],
    settings: dict[str, Any],
    cache: ProbeCache,
) -> dict[str, Any]:
    """Check whether a concept's decline also holds in the external index.

    A concept that keeps growing outside the screened corpus is a corpus
    artefact, not a stalled research line.
    """
    phrases = entity_phrases(candidate, candidate.get("concept", ""))
    if not phrases:
        return {"verdict": FAILED, "probes": [], "reason": "missing concept label"}

    lookback = max(int(settings.get("temporal_lookback_years", 2)), 1)
    end_year = candidate.get("analysis_end_year")
    probes = []
    external_decay: float | None = None
    for source, search in searchers.items():
        for query in single_queries(phrases)[: int(settings.get("max_queries_per_candidate", 4))]:
            probe = run_probe(source, search, query, [phrases], 0.0, cache, mode="year_counts")
            probes.append(probe)
            counts = {int(k): int(v) for k, v in (probe.get("year_counts") or {}).items()}
            if not probe.get("ok") or not counts:
                continue
            final_year = int(end_year) if end_year else max(counts)
            recent = sum(counts.get(y, 0) for y in range(final_year - lookback + 1, final_year + 1))
            earlier = sum(
                counts.get(y, 0) for y in range(final_year - 2 * lookback + 1, final_year - lookback + 1)
            )
            observed = 1.0 - recent / earlier if earlier else 0.0
            external_decay = observed if external_decay is None else min(external_decay, observed)

    if not probes or any(not probe.get("ok") for probe in probes):
        return {"verdict": FAILED, "probes": probes, "external_decay_rate": external_decay}
    threshold = float(settings.get("temporal_decay_threshold", 0.30))
    if external_decay is None:
        return {"verdict": FAILED, "probes": probes, "external_decay_rate": None}
    if external_decay < threshold:
        return {
            "verdict": REFUTED,
            "probes": probes,
            "external_decay_rate": round(external_decay, 4),
            "reason": "external publication activity does not decline",
        }
    return {
        "verdict": CORROBORATED,
        "probes": probes,
        "external_decay_rate": round(external_decay, 4),
    }


def _verify_orphan_cluster(
    candidate: dict[str, Any],
    searchers: dict[str, Callable[..., dict[str, Any]]],
    settings: dict[str, Any],
    cache: ProbeCache,
) -> dict[str, Any]:
    """Probe whether the cluster's own concepts are co-studied externally.

    Isolation inside this graph is only meaningful if the cluster's leading
    concepts are also rarely joined in the wider literature.
    """
    members = [canonical_entity_label(value) for value in candidate.get("members", [])]
    members = [value for value in members if value]
    if len(members) < 2:
        return {"verdict": NOT_APPLICABLE, "probes": [], "reason": "cluster too small to probe"}

    probe_depth = max(int(settings.get("orphan_probe_pairs", 3)), 1)
    coverage = float(settings.get("co_mention_token_coverage", 0.60))
    pairs = [(members[0], other) for other in members[1 : probe_depth + 1]]

    probes = []
    for source, search in searchers.items():
        for left, right in pairs:
            probes.append(
                run_probe(
                    source,
                    search,
                    f'("{left}" AND "{right}")',
                    [[left], [right]],
                    coverage,
                    cache,
                )
            )
    verdict, refuting = _verdict_from_probes(probes)
    return {"verdict": verdict, "probes": probes, "refuting_papers": refuting}


def _verify_evidence_gap(
    candidate: dict[str, Any],
    searchers: dict[str, Callable[..., dict[str, Any]]],
    settings: dict[str, Any],
    cache: ProbeCache,
) -> dict[str, Any]:
    subj = candidate.get("subject") or candidate.get("head") or ""
    cap = candidate.get("missing_capability") or candidate.get("tail") or ""
    subj_phrases = entity_phrases(candidate, subj)
    cap_phrases = entity_phrases(candidate, cap)
    if not subj_phrases or not cap_phrases:
        return {"verdict": FAILED, "probes": [], "reason": "missing subject or missing_capability"}

    coverage = float(settings.get("co_mention_token_coverage", 0.60))
    max_queries = int(settings.get("max_queries_per_candidate", 4))
    entity_groups = [subj_phrases, cap_phrases]
    queries = pair_queries(subj_phrases, cap_phrases)[:max_queries]

    probes = []
    for source, search in searchers.items():
        for query in queries:
            probes.append(
                run_probe(source, search, query, entity_groups, coverage, cache)
            )
    verdict, refuting = _verdict_from_probes(probes)
    return {"verdict": verdict, "probes": probes, "refuting_papers": refuting}


def verify_candidate(
    candidate: dict[str, Any],
    searchers: dict[str, Callable[..., dict[str, Any]]],
    settings: dict[str, Any],
    cache: ProbeCache | None = None,
) -> dict[str, Any]:
    """Verify one candidate's absence claim against external indexes."""
    cache = cache if cache is not None else ProbeCache(None)
    if not searchers:
        return {
            "verdict": FAILED,
            "probes": [],
            "reason": "no external verification source configured",
            "sources": [],
        }

    handlers = {
        "evidence_gap": _verify_evidence_gap,
        "missing_link": _verify_missing_link,
        "temporal_decay": _verify_temporal_decay,
        "orphan_cluster": _verify_orphan_cluster,
    }
    handler = handlers.get(str(candidate.get("type")))
    if handler is None:
        return {"verdict": NOT_APPLICABLE, "probes": [], "sources": sorted(searchers)}

    result = handler(candidate, searchers, settings, cache)
    result["sources"] = sorted(searchers)
    result["probe_count"] = len(result.get("probes", []))
    return result


def verify_all_candidates(
    config: dict[str, Any],
    searchers: dict[str, Callable[..., dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Stage entry point: verify every post-gate candidate against indexes.

    Only candidates that survived the local gate are probed, because probing is
    network-bound and rejected candidates cannot become gaps.
    """
    settings = config.get("external_verification", {}) or {}
    output_dir = ensure_dir(config["paths"]["outputs"])
    review_path = output_dir / "review_required_gaps.json"
    eligible_path = output_dir / "evidence_clear_candidates.json"
    if not review_path.exists() and not eligible_path.exists():
        raise FileNotFoundError("Run the validate stage before external verification")

    if not settings.get("enabled", True):
        logger.warning("External verification disabled in configuration; nothing probed")
        save_json(
            {"summary": {"enabled": False, "verified": 0}, "candidates": []},
            output_dir / "external_verification.json",
        )
        return {"enabled": False, "verified": 0}

    if searchers is None:
        searchers = build_searchers(settings)
    cache = ProbeCache(output_dir / "external_probe_cache.json")
    checkpoint_path = output_dir / "external_verification_checkpoint.json"
    checkpoint_data: dict[str, Any] = {}
    if checkpoint_path.exists():
        try:
            loaded_cp = load_json(checkpoint_path)
            if isinstance(loaded_cp, dict):
                checkpoint_data = loaded_cp
                logger.info("Resuming external verification from checkpoint: %d done", len(checkpoint_data))
        except Exception:
            pass

    queues: list[tuple[str, dict[str, Any]]] = []
    for label, path in (("review_required", review_path), ("automatically_eligible", eligible_path)):
        if not path.exists():
            continue
        for category, candidates in (load_json(path) or {}).items():
            for candidate in candidates:
                queues.append((label, {**candidate, "_queue": label, "_category": category}))

    from src.gap_provenance import candidate_identity

    records = []
    counts = {REFUTED: 0, CORROBORATED: 0, FAILED: 0, NOT_APPLICABLE: 0}
    for _, candidate in queues:
        c_key = candidate_identity(candidate)
        if c_key in checkpoint_data:
            rec = checkpoint_data[c_key]
            verification = rec.get("external_verification", {})
        else:
            verification = verify_candidate(candidate, searchers, settings, cache)
            cache.flush()
            rec = {
                "type": candidate.get("type"),
                "queue": candidate.get("_queue"),
                "category": candidate.get("_category"),
                "head": candidate.get("head"),
                "tail": candidate.get("tail"),
                "relation": candidate.get("relation"),
                "concept": candidate.get("concept"),
                "community_id": candidate.get("community_id"),
                "subject": candidate.get("subject"),
                "missing_capability": candidate.get("missing_capability"),
                "evidence_cell_id": candidate.get("evidence_cell_id"),
                "external_verification": verification,
            }
            checkpoint_data[c_key] = rec
            save_json(checkpoint_data, checkpoint_path)

        verdict = verification.get("verdict", FAILED)
        counts[verdict] = counts.get(verdict, 0) + 1
        records.append(rec)

    cache.flush()

    summary = {
        "enabled": True,
        "sources": sorted(searchers),
        "verified": len(records),
        "verdicts": counts,
        "policy": "refuted removes a candidate; failed probes never corroborate absence",
    }
    save_json(
        {"summary": summary, "candidates": records},
        output_dir / "external_verification.json",
    )
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass
    logger.info("External verification: %s", summary)
    return summary


# Public alias for pipeline runner
verify_against_external_indices = verify_all_candidates


if __name__ == "__main__":
    import yaml

    with open("config.yaml", encoding="utf-8") as stream:
        verify_all_candidates(yaml.safe_load(stream))
