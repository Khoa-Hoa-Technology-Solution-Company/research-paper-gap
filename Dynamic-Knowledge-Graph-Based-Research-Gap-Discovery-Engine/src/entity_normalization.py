"""Deterministic entity-label normalization shared across pipeline stages."""

from __future__ import annotations

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
