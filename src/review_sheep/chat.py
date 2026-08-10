"""Continuous CLI conversation with the intent-routed LangGraph chatbot."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any, TextIO, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from review_sheep.chat_graph import ChatState, create_chatbot_graph
from review_sheep.checkout import GitCheckoutSource
from review_sheep.github import GitHubPullRequestReader
from review_sheep.inquiry import create_inquiry_agent
from review_sheep.intent import create_intent_classifier
from review_sheep.providers import ModelFactory, github_client, openai_model
from review_sheep.review import create_deep_review_agent


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
) -> int:
    """Read .env and continuously route Inquiry and Review requests."""
    output = output or sys.stdout
    error = error or sys.stderr
    load_dotenv(dotenv_path=".env")

    try:
        token = _required_env("GITHUB_TOKEN")
        model_name = _required_env("OPENAI_MODEL")
        api_key = _required_env("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL", "").strip() or None
    except RuntimeError as config_error:
        print(f"error: {config_error}", file=error)
        return 2

    try:
        client = github_factory(token)
    except Exception as github_error:  # noqa: BLE001 - concise CLI boundary
        print(
            f"error: {type(github_error).__name__}: {github_error}",
            file=error,
        )
        return 1

    try:
        github = GitHubPullRequestReader(client=client)
        try:
            model = model_factory(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
            )
        except (RuntimeError, ValueError) as config_error:
            print(f"error: {config_error}", file=error)
            return 2

        inquiry_agent = create_inquiry_agent(model=model, github=github)
        intent_classifier = create_intent_classifier(model=model)
        checkout = GitCheckoutSource(
            revisions=github,
            root=os.getenv("REVIEW_CHECKOUT", ".").strip() or ".",
        )
        review_agent = create_deep_review_agent(source=checkout, model=model)
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
                conversation["messages"] = [
                    *conversation["messages"],
                    HumanMessage(content=user_message),
                ]
                conversation = cast(
                    ChatState,
                    chatbot.invoke(conversation),
                )
                print(
                    f"Bot: {conversation['messages'][-1].content}",
                    file=output,
                )
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
        client.close()
