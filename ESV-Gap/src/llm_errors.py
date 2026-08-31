"""Shared fail-fast classification for OpenAI-compatible LLM API errors."""

from __future__ import annotations

from typing import Any


class LLMAuthenticationError(RuntimeError):
    """Raised when an LLM provider rejects the configured API credential."""


class LLMRateLimitError(RuntimeError):
    """Raised after every configured provider key is currently rate-limited."""


def is_authentication_error(exc: BaseException) -> bool:
    """Recognize authentication failures across SDK versions/providers."""
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    body: Any = getattr(exc, "body", None)
    error_code = ""
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            error_code = str(error.get("code", ""))
    text = f"{error_code} {exc}".casefold()
    return str(status_code) in {"401", "403"} or any(marker in text for marker in (
        "invalid_api_key",
        "invalid api key",
        "incorrect api key",
        "authentication_error",
        "unauthorized",
    ))


def authentication_error_message() -> str:
    return (
        "Groq rejected every available API key (401/403). Replace the keys in "
        "the sidebar or set valid GROQ_API_KEYS/GROQ_API_KEY values, then resume. The relevance "
        "threshold is not the cause of this error."
    )


def rate_limit_error_message() -> str:
    return (
        "Every configured Groq API key is currently rate-limited or out of quota. "
        "Progress was preserved. Add another key or resume after quota renewal."
    )
