"""In-memory Groq API-key rotation for OpenAI-compatible clients.

Keys are deliberately never persisted by this module.  A request is retried
with the next configured key when Groq reports rate limiting, exhausted quota,
or an invalid credential.  If every key fails, the final provider exception is
re-raised so the existing checkpoint/resume logic can pause safely.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable, Iterable, Sequence

from openai import OpenAI

from src.llm_errors import is_authentication_error


logger = logging.getLogger(__name__)


def _split_key_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;\r\n]+", value) if item.strip()]
    if isinstance(value, Iterable):
        keys: list[str] = []
        for item in value:
            keys.extend(_split_key_value(item))
        return keys
    return [str(value).strip()] if str(value).strip() else []


def normalise_groq_keys(values: object = None, include_environment: bool = True) -> list[str]:
    """Return de-duplicated Groq keys while preserving their priority order."""
    candidates = _split_key_value(values)
    if include_environment:
        candidates.extend(_split_key_value(os.getenv("GROQ_API_KEYS")))
        candidates.extend(_split_key_value(os.getenv("GROQ_API_KEY")))
        candidates.extend(_split_key_value(os.getenv("OPENAI_API_KEY")))

    result: list[str] = []
    seen: set[str] = set()
    for key in candidates:
        if key and key not in seen:
            result.append(key)
            seen.add(key)
    return result


def groq_keys_from_config(config: dict | None = None, explicit_keys: object = None) -> list[str]:
    api_keys = (config or {}).get("api_keys", {}) or {}
    configured = [
        explicit_keys,
        api_keys.get("groq_keys"),
        api_keys.get("groq"),
    ]
    return normalise_groq_keys(configured, include_environment=True)


def is_rate_limit_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    text = str(exc).casefold()
    return str(status_code) == "429" or any(marker in text for marker in (
        "rate_limit",
        "rate limit",
        "tokens per day",
        "tokens per minute",
        "requests per day",
        "requests per minute",
        "quota",
        " tpd",
        " tpm",
        " rpd",
        " rpm",
    ))


def _is_long_lived_failure(exc: BaseException) -> bool:
    text = str(exc).casefold()
    return is_authentication_error(exc) or any(marker in text for marker in (
        "tokens per day",
        "requests per day",
        "daily quota",
        " tpd",
        " rpd",
    ))


def masked_key_label(key: str, position: int) -> str:
    suffix = key[-4:] if len(key) >= 4 else "****"
    return f"key #{position + 1} (…{suffix})"


class _CompletionsProxy:
    def __init__(self, owner: "RotatingGroqClient") -> None:
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._create(**kwargs)


class _ChatProxy:
    def __init__(self, owner: "RotatingGroqClient") -> None:
        self.completions = _CompletionsProxy(owner)


class RotatingGroqClient:
    """Small compatibility wrapper exposing ``client.chat.completions.create``."""

    def __init__(
        self,
        keys: Sequence[str],
        *,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 30.0,
        max_retries: int = 0,
        client_factory: Callable[..., object] = OpenAI,
    ) -> None:
        self.keys = normalise_groq_keys(keys, include_environment=False)
        if not self.keys:
            raise ValueError(
                "Missing Groq API key. Enter at least one key in the app or set "
                "GROQ_API_KEYS/GROQ_API_KEY."
            )
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._client_factory = client_factory
        self._clients: dict[int, object] = {}
        self._disabled: set[int] = set()
        self._index = 0
        self._last_pool_error: BaseException | None = None
        self.chat = _ChatProxy(self)

    @property
    def pool_size(self) -> int:
        return len(self.keys)

    @property
    def active_key_number(self) -> int:
        return self._index + 1

    def _client(self, index: int):
        if index not in self._clients:
            self._clients[index] = self._client_factory(
                api_key=self.keys[index],
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._clients[index]

    def _candidate_indexes(self) -> list[int]:
        size = len(self.keys)
        return [
            (self._index + offset) % size
            for offset in range(size)
            if (self._index + offset) % size not in self._disabled
        ]

    def _create(self, **kwargs):
        indexes = self._candidate_indexes()
        if not indexes:
            if self._last_pool_error is not None:
                raise self._last_pool_error
            raise RuntimeError("All Groq API keys in this in-memory pool are unavailable.")

        last_error: BaseException | None = None
        for attempt, index in enumerate(indexes, start=1):
            self._index = index
            try:
                return self._client(index).chat.completions.create(**kwargs)
            except Exception as exc:
                if not (is_rate_limit_error(exc) or is_authentication_error(exc)):
                    raise
                last_error = exc
                self._last_pool_error = exc
                if _is_long_lived_failure(exc):
                    self._disabled.add(index)
                remaining = len(indexes) - attempt
                if remaining:
                    next_index = indexes[attempt]
                    logger.warning(
                        "Groq %s was limited or rejected; switching automatically "
                        "to %s (%d key(s) remain for this request).",
                        masked_key_label(self.keys[index], index),
                        masked_key_label(self.keys[next_index], next_index),
                        remaining,
                    )
                    self._index = next_index
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Groq key rotation failed without a provider response.")


def create_groq_client(
    config: dict | None = None,
    explicit_keys: object = None,
    *,
    timeout: float = 30.0,
    max_retries: int = 0,
) -> RotatingGroqClient:
    return RotatingGroqClient(
        groq_keys_from_config(config, explicit_keys),
        timeout=timeout,
        max_retries=max_retries,
    )
