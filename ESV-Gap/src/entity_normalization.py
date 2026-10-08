"""Deterministic entity-label normalization shared across pipeline stages."""

from __future__ import annotations

import math
import re
import unicodedata


_TYPE_SUFFIX = re.compile(
    r"\s*\[\s*(?:method|dataset|metric|concept|finding|tool|unknown)\s*\]\s*$",
    flags=re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


def canonical_entity_label(value: object) -> str:
    """Remove extractor type suffixes while preserving a readable label.

    Entity type is graph metadata, not part of entity identity.  Keeping labels
    such as ``JWT [METHOD]`` in the node name caused coverage retrieval to
    require the word ``method`` and allowed case/type variants to form
    artificial missing links.
    """
    label = unicodedata.normalize("NFKC", str(value or ""))
    previous = None
    while label != previous:
        previous = label
        label = _TYPE_SUFFIX.sub("", label)
    return _WHITESPACE.sub(" ", label).strip()


def canonical_entity_key(value: object) -> str:
    """Return the case-insensitive identity key for an entity label."""
    return canonical_entity_label(value).casefold()


STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "using", "via", "with",
}


def entity_tokens(text: object) -> set[str]:
    """Tokenize an entity label or document for lexical co-mention matching.

    Numeric suffixes also emit a versionless form so that protocol variants
    such as ``OAuth2`` match ``OAuth``.  Shared by the local coverage screen
    and by external absence verification, which must agree on what counts as
    a mention.
    """
    output: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", canonical_entity_label(text).lower()):
        if token in STOPWORDS:
            continue
        output.add(token)
        versionless = re.sub(r"\d+$", "", token)
        if versionless and versionless != token and versionless not in STOPWORDS:
            output.add(versionless)
    return output


def mentions_entity(text_tokens: set[str], entity_tokens_set: set[str], coverage: float) -> bool:
    """Return whether enough of an entity's tokens appear in a document."""
    if not entity_tokens_set:
        return False
    required = max(1, math.ceil(len(entity_tokens_set) * coverage))
    return len(entity_tokens_set.intersection(text_tokens)) >= required
