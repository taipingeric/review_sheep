from __future__ import annotations

import pytest

from review_sheep.config import LangfuseConfig, MLflowConfig
from review_sheep.tracing import (
    create_langfuse_handler,
    create_mlflow_handler,
    create_tracing_handlers,
    langfuse_turn_config,
    tracing_turn_config,
)


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
        "review_sheep_session_id": "session-123",
        "review_sheep_tags": ["review-sheep", "chat", "langgraph"],
        "langfuse_session_id": "session-123",
        "langfuse_tags": ["review-sheep", "chat", "langgraph"],
        "review_sheep_turn": 7,
    }


def test_mlflow_is_optional_when_configuration_is_absent() -> None:
    assert create_mlflow_handler(None) is None


def test_mlflow_configuration_requires_tracking_uri() -> None:
    with pytest.raises(ValueError, match="tracking_uri is required"):
        MLflowConfig(tracking_uri="")


def test_configured_tracing_backends_share_one_turn_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    langfuse_handler = object()
    mlflow_handler = object()
    monkeypatch.setattr(
        "review_sheep.tracing.create_langfuse_handler",
        lambda _: langfuse_handler,
    )
    monkeypatch.setattr(
        "review_sheep.tracing.create_mlflow_handler",
        lambda _: mlflow_handler,
    )

    handlers = create_tracing_handlers(
        langfuse=LangfuseConfig(public_key="pk", secret_key="sk"),
        mlflow=MLflowConfig(tracking_uri="http://mlflow"),
    )
    config = tracing_turn_config(
        handlers=handlers,
        session_id="session-123",
        turn=7,
    )

    assert handlers == [langfuse_handler, mlflow_handler]
    assert config is not None
    assert config["callbacks"] == [langfuse_handler, mlflow_handler]
    assert config["run_name"] == "review-sheep-chat-turn"
    assert config["metadata"]["review_sheep_session_id"] == "session-123"
    assert config["metadata"]["langfuse_session_id"] == "session-123"
    assert config["metadata"]["review_sheep_turn"] == 7


def test_tracing_backend_failure_does_not_disable_other_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlflow_handler = object()
    monkeypatch.setattr(
        "review_sheep.tracing.create_langfuse_handler",
        lambda _: (_ for _ in ()).throw(RuntimeError("Langfuse unavailable")),
    )
    monkeypatch.setattr(
        "review_sheep.tracing.create_mlflow_handler",
        lambda _: mlflow_handler,
    )

    handlers = create_tracing_handlers(
        langfuse=LangfuseConfig(public_key="pk", secret_key="sk"),
        mlflow=MLflowConfig(tracking_uri="http://mlflow"),
    )

    assert handlers == [mlflow_handler]
