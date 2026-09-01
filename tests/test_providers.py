from __future__ import annotations

import sys
import types
from typing import Any, cast

import pytest
from pydantic import SecretStr

from review_sheep import providers as providers_module
from review_sheep.providers import (
    anthropic_model,
    github_client,
    openai_model,
    parse_anthropic_custom_headers,
    resolve_anthropic_timeout,
)


def test_github_client_constructs_a_token_authenticated_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAuth:
        @staticmethod
        def Token(token: str) -> tuple[str, str]:
            captured["token"] = token
            return ("token-auth", token)

    class FakeGithub:
        def __init__(self, *, auth: object) -> None:
            captured["auth"] = auth

    monkeypatch.setattr(providers_module, "Auth", FakeAuth)
    monkeypatch.setattr(providers_module, "Github", FakeGithub)

    client = github_client("github-secret")

    assert isinstance(client, FakeGithub)
    assert captured == {
        "token": "github-secret",
        "auth": ("token-auth", "github-secret"),
    }


def test_openai_model_constructs_a_zero_temperature_model_without_exposing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def __repr__(self) -> str:
            return repr(captured)

        __str__ = __repr__

    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChatOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    model = openai_model(
        model="gpt-injected",
        api_key="openai-secret",
        base_url=None,
    )

    assert isinstance(model, FakeChatOpenAI)
    assert captured["model"] == "gpt-injected"
    assert captured["base_url"] is None
    assert captured["temperature"] == 0
    assert captured["use_responses_api"] is True
    api_key = captured["api_key"]
    assert isinstance(api_key, SecretStr)
    assert api_key.get_secret_value() == "openai-secret"
    assert "openai-secret" not in repr(api_key)
    assert "openai-secret" not in repr(model)
    assert "openai-secret" not in str(model)


def test_openai_model_raises_a_friendly_error_when_the_package_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "langchain_openai", None)

    with pytest.raises(RuntimeError, match="uv sync --extra openai") as error:
        openai_model(
            model="gpt-injected",
            api_key="openai-secret",
            base_url=None,
        )

    assert "openai-secret" not in str(error.value)


def test_anthropic_custom_headers_use_claude_code_format() -> None:
    assert parse_anthropic_custom_headers(
        "X-Team: review-sheep\nAuthorization: Custom value\n\n"
    ) == {
        "X-Team": "review-sheep",
        "Authorization": "Custom value",
    }


@pytest.mark.parametrize("value", ["missing-colon", ": value", "Name:"])
def test_anthropic_custom_headers_reject_malformed_lines(value: str) -> None:
    with pytest.raises(ValueError, match="Name: Value"):
        parse_anthropic_custom_headers(value)


def test_anthropic_model_passes_bearer_configuration_to_provider_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAnthropicModel:
        @classmethod
        def model_validate(cls, values: dict[str, Any]) -> FakeAnthropicModel:
            captured.update(values)
            return cls()

    monkeypatch.setattr(
        providers_module,
        "_get_auth_token_chat_anthropic_cls",
        lambda: FakeAnthropicModel,
    )

    model = anthropic_model(
        model="gateway-sonnet",
        auth_token="auth-token",
        base_url="https://gateway.example.com",
        custom_headers="X-Team: review-sheep",
        request_timeout=30,
    )

    assert isinstance(model, FakeAnthropicModel)
    assert captured["model"] == "gateway-sonnet"
    assert captured["auth_token"].get_secret_value() == "auth-token"
    assert captured["base_url"] == "https://gateway.example.com"
    assert captured["default_headers"] == {"X-Team": "review-sheep"}
    assert captured["default_request_timeout"] == 30
    assert captured["temperature"] == 0


def test_anthropic_model_redacts_custom_header_values_from_representations() -> None:
    model = anthropic_model(
        model="gateway-sonnet",
        auth_token="auth-token",
        base_url="https://gateway.example.com",
        custom_headers="Authorization: Bearer header-secret\nX-Team: review-sheep",
    )

    assert "header-secret" not in repr(model)
    assert "header-secret" not in str(model)
    assert "auth-token" not in repr(model)
    assert "auth-token" not in str(model)
    assert "Authorization" in repr(model)
    headers = cast(Any, model).default_headers
    assert headers["Authorization"] == "Bearer header-secret"


@pytest.mark.parametrize(
    ("timeout", "expected"),
    [
        (None, (10, 600, 600, 10)),
        (30, 30),
    ],
)
def test_anthropic_timeout_resolution_supports_gateway_fallback_and_explicit_values(
    timeout: float | None, expected: float | tuple[int, ...]
) -> None:
    assert resolve_anthropic_timeout(timeout) == expected


def test_anthropic_model_reuses_the_same_bearer_auth_class_across_calls() -> None:
    first = anthropic_model(
        model="gateway-sonnet",
        auth_token="auth-token",
        base_url="https://gateway.example.com",
        custom_headers="",
    )
    second = anthropic_model(
        model="gateway-sonnet",
        auth_token="auth-token",
        base_url="https://gateway.example.com",
        custom_headers="",
    )

    assert type(first) is type(second)


def test_anthropic_model_raises_a_friendly_error_when_the_package_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers_module._get_auth_token_chat_anthropic_cls.cache_clear()
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)

    with pytest.raises(RuntimeError, match="uv sync --extra anthropic") as error:
        anthropic_model(
            model="gateway-sonnet",
            auth_token="auth-token",
            base_url="https://gateway.example.com",
            custom_headers="",
        )

    assert "auth-token" not in str(error.value)
