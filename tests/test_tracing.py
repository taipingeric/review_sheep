from __future__ import annotations

from typing import Any

import pytest

from review_sheep.tracing import create_langfuse_handler, langfuse_turn_config


def _clear_langfuse(monkeypatch: Any) -> None:
    for name in (
        "LANGFUSE_TRACING_ENABLED",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_langfuse_is_optional_when_credentials_are_absent(monkeypatch: Any) -> None:
    _clear_langfuse(monkeypatch)

    assert create_langfuse_handler() is None


def test_langfuse_rejects_partial_credentials(monkeypatch: Any) -> None:
    _clear_langfuse(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")

    with pytest.raises(RuntimeError, match="must both be configured"):
        create_langfuse_handler()


def test_langfuse_handler_uses_sdk_v4_environment_configuration(
    monkeypatch: Any,
) -> None:
    _clear_langfuse(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:3000")

    handler = create_langfuse_handler()

    assert handler is not None
    assert handler.__class__.__module__.startswith("langfuse")


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
