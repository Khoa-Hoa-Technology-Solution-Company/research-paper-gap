from types import SimpleNamespace

import pytest

from src.groq_key_pool import (
    RotatingGroqClient,
    groq_keys_from_config,
    normalise_groq_keys,
)


class ProviderError(RuntimeError):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class FakeClientFactory:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def __call__(self, api_key, **_kwargs):
        factory = self

        class Completions:
            def create(self, **kwargs):
                factory.calls.append((api_key, kwargs))
                outcome = factory.outcomes[api_key].pop(0)
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

        return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def test_normalise_keys_deduplicates_and_accepts_multiline(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_env_1;gsk_env_2")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env_2")
    assert normalise_groq_keys("gsk_ui_1\ngsk_ui_1,gsk_ui_2") == [
        "gsk_ui_1",
        "gsk_ui_2",
        "gsk_env_1",
        "gsk_env_2",
    ]


def test_configured_key_list_precedes_legacy_and_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    keys = groq_keys_from_config({
        "api_keys": {"groq_keys": ["gsk_a", "gsk_b"], "groq": "gsk_legacy"}
    })
    assert keys == ["gsk_a", "gsk_b", "gsk_legacy", "gsk_env"]


def test_rate_limit_rotates_to_next_key_without_losing_request():
    expected = object()
    factory = FakeClientFactory({
        "gsk_one": [ProviderError("429 rate_limit", 429)],
        "gsk_two": [expected],
    })
    client = RotatingGroqClient(
        ["gsk_one", "gsk_two"], client_factory=factory, max_retries=0
    )

    result = client.chat.completions.create(model="test-model", messages=[])

    assert result is expected
    assert [call[0] for call in factory.calls] == ["gsk_one", "gsk_two"]
    assert client.active_key_number == 2


def test_invalid_key_is_disabled_after_failover():
    first = object()
    second = object()
    factory = FakeClientFactory({
        "gsk_bad": [ProviderError("invalid_api_key", 401)],
        "gsk_good": [first, second],
    })
    client = RotatingGroqClient(
        ["gsk_bad", "gsk_good"], client_factory=factory, max_retries=0
    )

    assert client.chat.completions.create(model="m") is first
    assert client.chat.completions.create(model="m") is second
    assert [call[0] for call in factory.calls] == ["gsk_bad", "gsk_good", "gsk_good"]


def test_non_quota_provider_error_does_not_rotate():
    factory = FakeClientFactory({
        "gsk_one": [ProviderError("model_not_found", 404)],
        "gsk_two": [object()],
    })
    client = RotatingGroqClient(["gsk_one", "gsk_two"], client_factory=factory)

    with pytest.raises(ProviderError, match="model_not_found"):
        client.chat.completions.create(model="missing")
    assert [call[0] for call in factory.calls] == ["gsk_one"]
