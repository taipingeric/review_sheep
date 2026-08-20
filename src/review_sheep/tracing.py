"""Optional Langfuse tracing for the interactive chatbot."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from review_sheep.config import LangfuseConfig

logger = logging.getLogger(__name__)


def create_langfuse_handler(config: LangfuseConfig | None) -> Any | None:
    """Create a LangChain callback from explicit Langfuse settings."""
    if config is None:
        logger.info("tracing.langfuse.disabled")
        return None

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=config.public_key,
        secret_key=config.secret_key,
        base_url=config.base_url,
        environment=config.environment,
    )
    handler = CallbackHandler(public_key=config.public_key)
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
    except Exception:  # tracing must not replace the chatbot's result
        logger.exception("tracing.langfuse.flush_failed")
