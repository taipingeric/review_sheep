from __future__ import annotations

import pytest

from review_sheep.config import LangfuseConfig
from review_sheep.tracing import create_langfuse_handler, langfuse_turn_config


def test_langfuse_is_optional_when_configuration_is_absent() -> None:
    assert create_langfuse_handler(None) is None


def test_langfuse_rejects_partial_credentials() -> None:
    with pytest.raises(ValueError, match="secret_key is required"):
        LangfuseConfig(public_key="pk-lf-test", secret_key="")


def test_langfuse_handler_uses_injected_configuration() -> None:
    handler = create_langfuse_handler(
        LangfuseConfig(
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
            base_url="http://127.0.0.1:3000",
            environment="test",
        )
    )

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
