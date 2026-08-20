from __future__ import annotations

import logging

import langfuse
import langfuse.langchain as langfuse_langchain
import pytest

from review_sheep.config import LangfuseConfig
from review_sheep.tracing import (
    create_langfuse_handler,
    flush_langfuse,
    langfuse_turn_config,
)


def test_langfuse_is_optional_when_credentials_are_absent() -> None:
    assert create_langfuse_handler(None) is None


def test_langfuse_rejects_partial_credentials() -> None:
    with pytest.raises(ValueError, match="secret_key is required"):
        LangfuseConfig(public_key="pk-lf-test", secret_key="")


def test_langfuse_handler_uses_explicit_configuration_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    handler = create_langfuse_handler(
        LangfuseConfig(
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
            base_url="http://127.0.0.1:3000",
        )
    )

    assert handler is not None
    assert handler.__class__.__module__.startswith("langfuse")
    assert handler._langfuse_client._tracing_enabled is True


def test_langfuse_handler_receives_injected_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs

    class FakeCallbackHandler:
        def __init__(self, **kwargs: object) -> None:
            captured["handler"] = kwargs

    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    monkeypatch.setattr(langfuse_langchain, "CallbackHandler", FakeCallbackHandler)

    handler = create_langfuse_handler(
        LangfuseConfig(
            public_key="pk-injected",
            secret_key="sk-injected",
            base_url="https://trace-injected.example",
            environment="ci",
        )
    )

    assert isinstance(handler, FakeCallbackHandler)
    assert captured == {
        "client": {
            "public_key": "pk-injected",
            "secret_key": "sk-injected",
            "base_url": "https://trace-injected.example",
            "environment": "ci",
            "tracing_enabled": True,
        },
        "handler": {"public_key": "pk-injected"},
    }


def test_langfuse_flush_does_not_log_sdk_error(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "langfuse-secret-injected"

    def failing_client(*, public_key: str | None = None) -> object:
        raise RuntimeError(secret)

    monkeypatch.setattr(langfuse, "get_client", failing_client)
    caplog.set_level(logging.ERROR)

    flush_langfuse(enabled=True, public_key="pk-lf-test")

    assert secret not in caplog.text
    assert "tracing.langfuse.flush_failed" in caplog.text


def test_langfuse_turn_config_groups_turns_into_one_session() -> None:
    handler = object()

    config = langfuse_turn_config(
        handler=handler,
        session_id="session-123",
        turn=7,
    )

    assert config is not None
    assert config["callbacks"] == [handler]
    assert config["run_name"] == "review-sheep-chat-turn"
    assert config["metadata"] == {
        "langfuse_session_id": "session-123",
        "langfuse_tags": ["review-sheep", "chat", "langgraph"],
        "review_sheep_turn": 7,
    }
