"""Provider construction shared by interactive and CI entrypoints."""

from __future__ import annotations

from typing import Protocol

from github import Auth, Github
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr


class ModelFactory(Protocol):
    def __call__(
        self, *, model: str, api_key: str, base_url: str | None
    ) -> str | BaseChatModel: ...


def github_client(token: str) -> Github:
    """Create the production read-only GitHub API client."""
    return Github(auth=Auth.Token(token))


def openai_model(*, model: str, api_key: str, base_url: str | None) -> BaseChatModel:
    """Create the production OpenAI-compatible LangChain model."""
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
