"""Small, dependency-free helpers for forward provenance records.

The project deliberately keeps these helpers separate from the historical
artifacts.  They are used only by future retrieval and extraction runs; they
must never be used to infer missing metadata for the legacy pilot.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


PROVENANCE_SCHEMA_VERSION = "kgtabi-provenance-v1"


def utc_now_iso() -> str:
    """Return an unambiguous UTC timestamp suitable for an audit record."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize JSON-like data deterministically before hashing it."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_query_key(query: str) -> str:
    """Deterministic, documented key used only to de-duplicate queries."""
    return re.sub(r"\s+", " ", query).strip().casefold()


def stable_chunk_id(
    paper_id: object,
    section_label: object,
    chunk_index: object,
    chunk_text: str,
) -> str:
    """Create an ID that changes when a supposedly identical chunk changes."""
    paper = str(paper_id or "unknown-paper")
    section = re.sub(r"[^A-Za-z0-9._-]+", "-", str(section_label or "unknown").strip())
    index = int(chunk_index) if str(chunk_index).isdigit() else str(chunk_index)
    return f"{paper}:{section}:{index}:{sha256_text(chunk_text)[:16]}"


def safe_filename(value: str, fallback: str = "record") -> str:
    """Keep a generated artifact path local and platform-safe."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned or fallback
