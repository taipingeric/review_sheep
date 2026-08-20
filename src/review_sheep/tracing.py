"""Optional Langfuse tracing for the interactive chatbot."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any

from langchain_core.runnables import RunnableConfig

from review_sheep.config import LangfuseConfig, MLflowConfig

logger = logging.getLogger(__name__)
_LANGFUSE_ENV_LOCK = Lock()


def create_langfuse_handler(config: LangfuseConfig | None) -> Any | None:
    """Create a LangChain callback when Langfuse is configured."""
    if config is None:
        logger.info("tracing.langfuse.disabled")
        return None

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    # Langfuse v4 ANDs its explicit flag with LANGFUSE_TRACING_ENABLED. Keep
    # that process-level switch from overriding the caller's explicit config.
    with _LANGFUSE_ENV_LOCK:
        previous_enabled = os.environ.get("LANGFUSE_TRACING_ENABLED")
        os.environ["LANGFUSE_TRACING_ENABLED"] = "true"
        try:
            Langfuse(
                public_key=config.public_key,
                secret_key=config.secret_key,
                base_url=config.base_url,
                environment=config.environment,
                tracing_enabled=True,
            )
            handler = CallbackHandler(public_key=config.public_key)
        finally:
            if previous_enabled is None:
                os.environ.pop("LANGFUSE_TRACING_ENABLED", None)
            else:
                os.environ["LANGFUSE_TRACING_ENABLED"] = previous_enabled
    logger.info(
        "tracing.langfuse.ready base_url=%s environment=%s",
        config.base_url,
        config.environment,
    )
    return handler


def create_mlflow_handler(config: MLflowConfig | None) -> Any | None:
    """Create an MLflow LangChain callback from explicit settings."""
    if config is None:
        logger.info("tracing.mlflow.disabled")
        return None

    import mlflow
    from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    handler = MlflowLangchainTracer()
    logger.info(
        "tracing.mlflow.ready experiment=%s",
        config.experiment_name,
    )
    return handler


def create_tracing_handlers(
    *,
    langfuse: LangfuseConfig | None,
    mlflow: MLflowConfig | None,
) -> list[Any]:
    """Create configured tracing callbacks without coupling backend failures."""
    handlers: list[Any] = []
    for name, factory, config in _tracing_backend_specs(
        langfuse=langfuse,
        mlflow=mlflow,
    ):
        handler = _create_tracing_handler(name=name, factory=factory, config=config)
        if handler is not None:
            handlers.append(handler)
    return handlers


def _tracing_backend_specs(
    *,
    langfuse: LangfuseConfig | None,
    mlflow: MLflowConfig | None,
) -> list[tuple[str, Callable[[Any], Any | None], Any]]:
    """Return only configured backend factories with one uniform callable type."""
    specs: list[tuple[str, Callable[[Any], Any | None], Any]] = []
    if langfuse is not None:
        specs.append(("langfuse", create_langfuse_handler, langfuse))
    if mlflow is not None:
        specs.append(("mlflow", create_mlflow_handler, mlflow))
    return specs


def _create_tracing_handler(
    *, name: str, factory: Callable[[Any], Any | None], config: Any
) -> Any | None:
    try:
        return factory(config)
    except Exception:  # noqa: BLE001 - tracing must not replace chatbot result
        logger.error("tracing.%s.initialization_failed", name)
        return None


def langfuse_turn_config(
    *, handler: Any | None, session_id: str, turn: int
) -> RunnableConfig | None:
    """Build one root LangGraph trace configuration for a chatbot turn."""
    if handler is None:
        return None
    return tracing_turn_config(handlers=[handler], session_id=session_id, turn=turn)


def tracing_turn_config(
    *, handlers: Sequence[Any], session_id: str, turn: int
) -> RunnableConfig | None:
    """Build one root LangGraph trace configuration for a Chat turn."""
    if not handlers:
        return None
    return RunnableConfig(
        callbacks=list(handlers),
        run_name="review-sheep-chat-turn",
        metadata={
            "review_sheep_session_id": session_id,
            "review_sheep_tags": ["review-sheep", "chat", "langgraph"],
            "langfuse_session_id": session_id,
            "langfuse_tags": ["review-sheep", "chat", "langgraph"],
            "review_sheep_turn": turn,
        },
    )


def ci_review_config(
    *, handlers: Sequence[Any], repo: str, number: int
) -> RunnableConfig | None:
    """Build one correlated root trace configuration for a CI Review."""
    if not handlers:
        return None
    session_id = f"ci:{repo}#{number}"
    tags = ["review-sheep", "ci", "review"]
    return RunnableConfig(
        callbacks=list(handlers),
        run_name="review-sheep-ci-review",
        tags=tags,
        metadata={
            "review_sheep_run_kind": "ci",
            "review_sheep_session_id": session_id,
            "review_sheep_tags": tags,
            "review_sheep_repo": repo,
            "review_sheep_pull_request": number,
            "langfuse_session_id": session_id,
            "langfuse_tags": tags,
        },
    )


def flush_langfuse(*, enabled: bool, public_key: str | None = None) -> None:
    """Flush queued spans before a short-lived CLI exits."""
    if not enabled:
        return
    try:
        from langfuse import get_client

        get_client(public_key=public_key).flush()
        logger.info("tracing.langfuse.flushed")
    except Exception:  # noqa: BLE001 - tracing must not replace chatbot result
        logger.error("tracing.langfuse.flush_failed")


def flush_mlflow(*, enabled: bool) -> None:
    """Flush queued MLflow spans before a short-lived CLI exits."""
    if not enabled:
        return
    try:
        import mlflow

        mlflow.flush_trace_async_logging()
        logger.info("tracing.mlflow.flushed")
    except Exception:  # noqa: BLE001 - tracing must not replace chatbot result
        logger.error("tracing.mlflow.flush_failed")


def flush_tracing(
    *,
    langfuse_enabled: bool,
    langfuse_public_key: str | None = None,
    mlflow_enabled: bool,
) -> None:
    """Flush all configured tracing backends independently."""
    flush_langfuse(
        enabled=langfuse_enabled,
        public_key=langfuse_public_key,
    )
    flush_mlflow(enabled=mlflow_enabled)
