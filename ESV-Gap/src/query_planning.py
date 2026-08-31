"""Safe, recall-oriented scholarly query planning."""

from __future__ import annotations

import re
from typing import Iterable


_BOOLEAN_PATTERN = re.compile(r'(?i)(?:"|\(|\)|\bAND\b|\bOR\b|\bNOT\b)')
_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "for", "from", "in", "of", "on", "or", "the",
    "to", "using", "with",
}


def query_words(value: str) -> list[str]:
    return [
        token.casefold() for token in _WORD_PATTERN.findall(str(value or ""))
        if token.casefold() not in _STOPWORDS
    ]


def is_safe_plain_query(query: str, topic: str = "") -> bool:
    """Accept only short keyword queries; reject Boolean/exact-match syntax."""
    if _BOOLEAN_PATTERN.search(str(query or "")):
        return False
    words = query_words(query)
    if not 2 <= len(words) <= 7:
        return False
    if topic:
        topic_words = set(query_words(topic))
        if len(topic_words.intersection(words)) < min(2, len(topic_words)):
            return False
    return True


def validate_generated_queries(
    queries: Iterable[str], topic: str, expected: int = 5
) -> list[str]:
    output = []
    seen = set()
    for query in queries:
        normalized = " ".join(str(query or "").split()).strip()
        key = normalized.casefold()
        if key in seen or not is_safe_plain_query(normalized, topic):
            continue
        output.append(normalized)
        seen.add(key)
    return output if len(output) == expected else []


def fallback_search_queries(topic: str) -> list[str]:
    """Return five deliberately broad query strata for a research topic."""
    lowered = str(topic or "").casefold()
    if "iot" in lowered and "intrusion" in lowered and "detect" in lowered:
        return [
            "deep learning IoT intrusion detection",
            "IoT intrusion detection review",
            "adversarial attacks IoT IDS",
            "IoT intrusion detection benchmark",
            "robust IoT intrusion detection",
        ]
    if "monolith" in lowered and any(
        value in lowered for value in ("security", "secure", "vulnerab")
    ):
        return [
            "monolithic software architecture security",
            "monolith microservices security review",
            "monolithic application vulnerabilities",
            "monolith security empirical evaluation",
            "monolithic architecture security challenges",
        ]

    words = list(dict.fromkeys(query_words(topic)))
    core = words[:5] or ["research", "topic"]
    anchor = core[-3:] if len(core) >= 3 else core
    plans = [
        core,
        anchor + ["review"],
        anchor + ["limitations"],
        anchor + ["benchmark"],
        anchor + ["challenges"],
    ]
    queries = []
    for plan in plans:
        query = " ".join(plan[:7])
        if query.casefold() not in {item.casefold() for item in queries}:
            queries.append(query)
    return queries[:5]


def broadening_queries(topic: str, existing: Iterable[str]) -> list[str]:
    existing_keys = {" ".join(str(query).split()).casefold() for query in existing}
    return [
        query for query in fallback_search_queries(topic)
        if query.casefold() not in existing_keys
    ]
