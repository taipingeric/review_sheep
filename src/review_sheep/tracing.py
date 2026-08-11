"""Optional Langfuse tracing for the interactive chatbot."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

_FALSE_VALUES = {"0", "false", "no", "off"}


def create_langfuse_handler() -> Any | None:
    """Create a LangChain callback when Langfuse is configured."""
    enabled = os.getenv("LANGFUSE_TRACING_ENABLED", "").strip().lower()
    if enabled in _FALSE_VALUES:
        logger.info("tracing.langfuse.disabled")
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key and not secret_key and not enabled:
        logger.info("tracing.langfuse.not_configured")
        return None
    if not public_key or not secret_key:
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must both be configured"
        )

    from langfuse.langchain import CallbackHandler

    handler = CallbackHandler()
    logger.info(
        "tracing.langfuse.ready base_url=%s environment=%s",
        os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "default"),
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


def flush_langfuse(*, enabled: bool) -> None:
    """Flush queued spans before a short-lived CLI exits."""
    if not enabled:
        return
    try:
        from langfuse import get_client

        get_client().flush()
        logger.info("tracing.langfuse.flushed")
    except Exception:  # tracing must not replace the chatbot's result
        logger.exception("tracing.langfuse.flush_failed")
