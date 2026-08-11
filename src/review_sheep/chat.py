"""Continuous CLI conversation with the intent-routed LangGraph chatbot."""

from __future__ import annotations

import logging
import os
import sys
import uuid
from collections.abc import Callable
from typing import Any, TextIO, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from review_sheep.chat_graph import ChatState, create_chatbot_graph
from review_sheep.github import GitHubPullRequestReader
from review_sheep.inquiry import create_inquiry_agent
from review_sheep.intent import create_intent_classifier
from review_sheep.manifest import create_manifest_review_agent
from review_sheep.providers import ModelFactory, github_client, openai_model
from review_sheep.runtime_logging import configure_console_logging
from review_sheep.tracing import (
    create_langfuse_handler,
    flush_langfuse,
    langfuse_turn_config,
)

logger = logging.getLogger(__name__)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


def main(
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO | None = None,
    error: TextIO | None = None,
    github_factory: Callable[[str], Any] = github_client,
    model_factory: ModelFactory = openai_model,
    reviewer_factory: Callable[..., Any] = create_manifest_review_agent,
    tracing_factory: Callable[[], Any | None] = create_langfuse_handler,
    tracing_flush: Callable[..., None] = flush_langfuse,
) -> int:
    """Read .env and continuously route Inquiry and Review requests."""
    configure_logs = error is None
    output = output or sys.stdout
    error = error or sys.stderr
    if configure_logs:
        configure_console_logging(
            stream=error,
            level=os.getenv("REVIEW_LOG_LEVEL", "INFO"),
        )
    logger.info("chat.start entrypoint=scripts/review_chat.py")
    load_dotenv(dotenv_path=".env")

    try:
        token = _required_env("GITHUB_TOKEN")
        model_name = _required_env("OPENAI_MODEL")
        api_key = _required_env("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL", "").strip() or None
        logger.info(
            "chat.config model=%s base_url=%s manifest_mode=true",
            model_name,
            base_url or "provider-default",
        )
    except RuntimeError as config_error:
        print(f"error: {config_error}", file=error)
        return 2

    try:
        client = github_factory(token)
        logger.info("chat.github_client.ready")
    except Exception as github_error:  # noqa: BLE001 - concise CLI boundary
        print(
            f"error: {type(github_error).__name__}: {github_error}",
            file=error,
        )
        return 1

    trace_handler: Any | None = None
    try:
        github = GitHubPullRequestReader(client=client)
        try:
            model = model_factory(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
            )
            logger.info("chat.model.ready model=%s", model_name)
            trace_handler = tracing_factory()
        except (RuntimeError, ValueError) as config_error:
            print(f"error: {config_error}", file=error)
            return 2

        inquiry_agent = create_inquiry_agent(model=model, github=github)
        intent_classifier = create_intent_classifier(model=model)
        review_agent = reviewer_factory(source=github, model=model)
        logger.info("chat.agents.ready inquiry=true review=manifest")
        session_id = uuid.uuid4().hex
        if trace_handler is not None:
            logger.info("chat.tracing.enabled session_id=%s", session_id)
        chatbot = create_chatbot_graph(
            agent=inquiry_agent,
            classifier=intent_classifier,
            reviewer=review_agent,
        )
        conversation = cast(
            ChatState,
            chatbot.invoke(ChatState(messages=[])),
        )
        print(
            "Review Sheep chatbot ready; type exit or quit to stop.",
            file=output,
        )
        print(f"Bot: {conversation['messages'][-1].content}", file=output)

        turn = 0
        while True:
            try:
                user_message = input_fn("You: ").strip()
            except EOFError:
                break
            if user_message.lower() in {"exit", "quit"}:
                break
            if not user_message:
                continue

            try:
                turn += 1
                logger.info("chat.turn.start characters=%d", len(user_message))
                conversation["messages"] = [
                    *conversation["messages"],
                    HumanMessage(content=user_message),
                ]
                conversation = cast(
                    ChatState,
                    chatbot.invoke(
                        conversation,
                        config=langfuse_turn_config(
                            handler=trace_handler,
                            session_id=session_id,
                            turn=turn,
                        ),
                    ),
                )
                print(
                    f"Bot: {conversation['messages'][-1].content}",
                    file=output,
                )
                logger.info("chat.turn.complete")
            except Exception as inquiry_error:  # noqa: BLE001 - keep the loop alive
                print(
                    f"error: {type(inquiry_error).__name__}: {inquiry_error}",
                    file=error,
                )
        return 0
    except Exception as github_error:  # noqa: BLE001 - concise CLI boundary
        print(
            f"error: {type(github_error).__name__}: {github_error}",
            file=error,
        )
        return 1
    finally:
        tracing_flush(enabled=trace_handler is not None)
        client.close()
        logger.info("chat.stop github_client_closed=true")
