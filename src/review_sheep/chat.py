"""Continuous CLI conversation with the LangGraph Inquiry chatbot."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any, Protocol, TextIO, cast

from dotenv import load_dotenv
from github import Auth, Github
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from review_sheep.chat_graph import InquiryChatState, create_chatbot_graph
from review_sheep.github import GitHubPullRequestReader
from review_sheep.inquiry import create_inquiry_agent


class ModelFactory(Protocol):
    def __call__(
        self, *, model: str, api_key: str, base_url: str | None
    ) -> str | BaseChatModel: ...


def _github_client(token: str) -> Github:
    return Github(auth=Auth.Token(token))


def _openai_model(*, model: str, api_key: str, base_url: str | None) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise RuntimeError(
            "langchain-openai is not installed; run uv sync --extra openai"
        ) from error

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0,
        use_responses_api=True,
    )


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
    github_factory: Callable[[str], Any] = _github_client,
    model_factory: ModelFactory = _openai_model,
) -> int:
    """Read .env and continuously answer Inquiry questions."""
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
        chatbot = create_chatbot_graph(agent=inquiry_agent)
        conversation = cast(
            InquiryChatState,
            chatbot.invoke(InquiryChatState(messages=[])),
        )
        print(
            "Inquiry chatbot ready; type exit or quit to stop.",
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
                    InquiryChatState,
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
