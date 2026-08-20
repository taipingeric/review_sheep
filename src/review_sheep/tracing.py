"""Optional Langfuse tracing for the interactive chatbot."""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

from langchain_core.runnables import RunnableConfig

from review_sheep.config import LangfuseConfig

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


def langfuse_turn_config(
    *, handler: Any | None, session_id: str, turn: int
) -> RunnableConfig | None:
    """Build one root LangGraph trace configuration for a chatbot turn."""
    if handler is None:
        return None
    return RunnableConfig(
        callbacks=[handler],
        run_name="review-sheep-chat-turn",
        metadata={
            "langfuse_session_id": session_id,
            "langfuse_tags": ["review-sheep", "chat", "langgraph"],
            "review_sheep_turn": turn,
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
