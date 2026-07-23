"""LLM access with forward-only, privacy-aware provenance logging.

``call_llm`` remains a string-returning compatibility wrapper.  New pipeline
stages should use ``call_llm_with_provenance`` and persist the returned call
record alongside their own output.  The log intentionally hashes prompts and
responses by default; it does not write API keys or source text to manifests.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any
from uuid import uuid4

from openai import OpenAI

from src import config
from src.provenance import PROVENANCE_SCHEMA_VERSION, safe_filename, sha256_json, sha256_text, utc_now_iso


@dataclass(frozen=True)
class LLMCallResult:
    """Text returned by the provider plus an immutable audit record."""

    text: str
    provenance: dict[str, Any]


class LLMCallError(RuntimeError):
    """An LLM failure whose non-secret provenance was still persisted."""

    def __init__(self, message: str, provenance: dict[str, Any]):
        super().__init__(message)
        self.provenance = provenance


def get_resolved_base_url() -> str:
    """Return the actual endpoint selected without exposing credentials."""
    if config.LLM_BASE_URL:
        return config.LLM_BASE_URL
    if config.LLM_PROVIDER == "groq":
        return "https://api.groq.com/openai/v1"
    return "https://api.openai.com/v1"


def get_llm_client() -> OpenAI:
    """Create an OpenAI-compatible client for the configured provider."""
    if not config.LLM_API_KEY:
        raise ValueError("API Key is missing! Please configure LLM_API_KEY in your .env file.")
    return OpenAI(api_key=config.LLM_API_KEY, base_url=get_resolved_base_url())


def get_llm_model() -> str:
    return config.LLM_MODEL


def configured_llm_identity() -> dict[str, Any]:
    """Return only non-secret configuration needed to interpret a future call."""
    return {
        "provider": config.LLM_PROVIDER,
        "model_identifier": config.LLM_MODEL,
        "configured_model_revision": config.LLM_MODEL_REVISION,
        "configured_model_release_date": config.LLM_MODEL_RELEASE_DATE,
        "base_url": get_resolved_base_url(),
        "configured_api_version": config.LLM_API_VERSION,
        "configured_api_date": config.LLM_API_DATE,
    }


def _retry_policy(max_retries: int) -> dict[str, Any]:
    return {
        "max_attempts": max_retries,
        "pre_request_delay_seconds": config.LLM_DELAY,
        "rate_limit_backoff": "provider-retry-after-plus-2s-or-exponential",
        "rate_limit_base_delay_seconds": config.LLM_RETRY_BASE_DELAY_SECONDS,
        "non_rate_limit_retry_delay_seconds": config.LLM_RETRY_NON_RATE_LIMIT_DELAY_SECONDS,
    }


def _append_call_log(record: dict[str, Any]) -> None:
    """Append a one-line, non-secret call record without changing old data."""
    os.makedirs(config.PROVENANCE_DIR, exist_ok=True)
    with open(config.LLM_CALL_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _provider_response_payload(response: Any, assistant_text: str) -> dict[str, Any]:
    """Return a JSON-safe provider response payload for hashing/storage.

    Prefer the SDK's JSON model dump when available.  A minimal fallback still
    records the observable provider metadata and assistant message without
    claiming it is a byte-for-byte HTTP transport capture.
    """
    if isinstance(response, dict):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json")
        if isinstance(payload, dict):
            return payload
    return {
        "id": _response_value(response, "id"),
        "model": _response_value(response, "model"),
        "created": _response_value(response, "created"),
        "system_fingerprint": _response_value(response, "system_fingerprint"),
        "assistant_message_content": assistant_text,
    }


def _store_response_if_permitted(call_id: str, payload: dict[str, Any]) -> str | None:
    if config.LLM_RAW_RESPONSE_POLICY != "store":
        return None
    os.makedirs(config.LLM_RESPONSE_DIR, exist_ok=True)
    filename = f"{safe_filename(call_id)}.json"
    path = os.path.join(config.LLM_RESPONSE_DIR, filename)
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "call_id": call_id,
        "stored_at_utc": utc_now_iso(),
        "provider_response_sha256": sha256_json(payload),
        # This is the SDK-normalized provider response payload when supported,
        # with a minimal documented fallback for compatible test/local clients.
        # Store it only when the rerun's sharing policy permits it.
        "provider_response": payload,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return os.path.relpath(path, config.DATA_DIR).replace(os.sep, "/")


def _is_rate_limit(error_text: str) -> bool:
    lowered = error_text.casefold()
    return any(marker in lowered for marker in ("429", "rate_limit", "quota", "resource_exhausted"))


def _next_retry_delay(error_text: str, attempt_index: int) -> float:
    retry_match = re.search(r"retry in ([\d.]+)s", error_text, flags=re.IGNORECASE)
    if retry_match:
        return float(retry_match.group(1)) + 2.0
    return config.LLM_RETRY_BASE_DELAY_SECONDS * (2 ** attempt_index)


def _response_value(response: Any, field: str) -> Any:
    """Read SDK objects and test doubles without assuming one concrete type."""
    if isinstance(response, dict):
        return response.get(field)
    return getattr(response, field, None)


def call_llm_with_provenance(
    prompt: str,
    *,
    temperature: float = 0.1,
    top_p: float | None = None,
    max_retries: int | None = None,
    prompt_version: str | None = None,
    purpose: str | None = None,
) -> LLMCallResult:
    """Call an LLM and return its text with reproducibility metadata.

    This records configured and provider-returned identifiers when available.
    It cannot manufacture a provider revision that the API does not expose;
    nullable fields make that limitation explicit in the manifest.
    """
    attempts_allowed = config.LLM_MAX_RETRIES if max_retries is None else max_retries
    if attempts_allowed < 1:
        raise ValueError("max_retries must be at least 1")
    effective_top_p = config.LLM_TOP_P if top_p is None else top_p
    if effective_top_p is not None and not 0 < effective_top_p <= 1:
        raise ValueError("top_p must be in (0, 1] when supplied")

    call_id = f"llm-{uuid4().hex}"
    started_at = utc_now_iso()
    record: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "record_type": "llm_call",
        "call_id": call_id,
        "purpose": purpose or "unspecified",
        "prompt_version": prompt_version,
        "prompt_sha256": sha256_text(prompt),
        "prompt_storage_policy": "hash-only",
        "request_started_at_utc": started_at,
        "llm": configured_llm_identity(),
        "generation_parameters": {
            "temperature": temperature,
            "top_p": effective_top_p,
        },
        "retry_policy": _retry_policy(attempts_allowed),
        "raw_response_storage_policy": config.LLM_RAW_RESPONSE_POLICY,
        "attempts": [],
        "status": "started",
    }

    try:
        client = get_llm_client()
    except Exception as exc:
        record.update({
            "status": "failed_before_request",
            "completed_at_utc": utc_now_iso(),
            "failure": {
                "error_type": type(exc).__name__,
                "error_sha256": sha256_text(str(exc)),
            },
        })
        _append_call_log(record)
        raise LLMCallError("Unable to initialize LLM client", record) from exc

    model = get_llm_model()
    last_error: Exception | None = None
    for attempt in range(attempts_allowed):
        attempt_record: dict[str, Any] = {"attempt": attempt + 1, "started_at_utc": utc_now_iso()}
        try:
            if config.LLM_DELAY:
                time.sleep(config.LLM_DELAY)
            request: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if effective_top_p is not None:
                request["top_p"] = effective_top_p
            response = client.chat.completions.create(**request)
            text = response.choices[0].message.content.strip()
            provider_payload = _provider_response_payload(response, text)
            response_hash = sha256_json(provider_payload) if config.LLM_RAW_RESPONSE_POLICY != "none" else None
            content_hash = sha256_text(text) if config.LLM_RAW_RESPONSE_POLICY != "none" else None
            response_path = None
            response_storage_error = None
            try:
                response_path = _store_response_if_permitted(call_id, provider_payload)
            except OSError as exc:
                # The provider already succeeded.  Retrying it would create a
                # duplicate paid request, so retain a hash-only record instead.
                response_storage_error = {
                    "error_type": type(exc).__name__,
                    "error_sha256": sha256_text(str(exc)),
                }
            attempt_record.update({"status": "success", "completed_at_utc": utc_now_iso()})
            record["attempts"].append(attempt_record)
            record.update({
                "status": "succeeded",
                "completed_at_utc": utc_now_iso(),
                "response": {
                    "raw_provider_response_sha256": response_hash,
                    "assistant_message_content_sha256": content_hash,
                    "storage_path": response_path,
                    "provider_response_id": _response_value(response, "id"),
                    "provider_model_identifier": _response_value(response, "model"),
                    "provider_created_at": _response_value(response, "created"),
                    "provider_system_fingerprint": _response_value(response, "system_fingerprint"),
                    "storage_error": response_storage_error,
                },
            })
            _append_call_log(record)
            return LLMCallResult(text=text, provenance=record)
        except Exception as exc:  # Provider exceptions have heterogeneous types.
            last_error = exc
            error_text = str(exc)
            rate_limited = _is_rate_limit(error_text)
            attempt_record.update({
                "status": "failed",
                "completed_at_utc": utc_now_iso(),
                "error_type": type(exc).__name__,
                "error_sha256": sha256_text(error_text),
                "rate_limit_like": rate_limited,
            })
            record["attempts"].append(attempt_record)
            print(f"[!] LLM API error (attempt {attempt + 1}/{attempts_allowed}): {exc}")
            if attempt == attempts_allowed - 1:
                break
            if rate_limited:
                sleep_time = _next_retry_delay(error_text, attempt)
                print(f"[*] Rate limit/quota signal. Sleeping {sleep_time:.2f}s before retrying.")
            else:
                sleep_time = config.LLM_RETRY_NON_RATE_LIMIT_DELAY_SECONDS
            if sleep_time:
                time.sleep(sleep_time)

    record.update({
        "status": "failed_after_retries",
        "completed_at_utc": utc_now_iso(),
        "failure": {
            "error_type": type(last_error).__name__ if last_error else "UnknownError",
            "error_sha256": sha256_text(str(last_error)) if last_error else None,
        },
    })
    _append_call_log(record)
    raise LLMCallError("LLM request failed after configured retries", record) from last_error


def call_llm(
    prompt: str,
    temperature: float = 0.1,
    max_retries: int | None = None,
    *,
    top_p: float | None = None,
    prompt_version: str | None = None,
    purpose: str | None = None,
) -> str:
    """Compatibility wrapper for existing callers that only require text."""
    return call_llm_with_provenance(
        prompt,
        temperature=temperature,
        top_p=top_p,
        max_retries=max_retries,
        prompt_version=prompt_version,
        purpose=purpose,
    ).text
