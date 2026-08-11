"""Provider construction shared by interactive and CI entrypoints."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import cached_property
from typing import Any, Protocol

from github import Auth, Github
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class ModelFactory(Protocol):
    def __call__(
        self, *, model: str, api_key: str, base_url: str | None
    ) -> str | BaseChatModel: ...


class AnthropicModelFactory(Protocol):
    def __call__(
        self,
        *,
        model: str,
        auth_token: str,
        base_url: str,
        custom_headers: str,
    ) -> str | BaseChatModel: ...


class _AuthTokenChatAnthropic(ChatAnthropic):
    """ChatAnthropic variant that uses the SDK's Bearer auth-token path."""

    auth_token: SecretStr

    @cached_property
    def _client_params(self) -> dict[str, Any]:
        params = dict(super()._client_params)
        params.pop("api_key", None)
        params["auth_token"] = self.auth_token.get_secret_value()
        return params


def github_client(token: str) -> Github:
    """Create the production read-only GitHub API client."""
    logger.info("provider.github.create auth=token")
    return Github(auth=Auth.Token(token))


def openai_model(*, model: str, api_key: str, base_url: str | None) -> BaseChatModel:
    """Create the production OpenAI-compatible LangChain model."""
    logger.info(
        "provider.openai.create model=%s base_url=%s",
        model,
        base_url or "provider-default",
    )
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


def anthropic_model(
    *,
    model: str,
    auth_token: str,
    base_url: str,
    custom_headers: str,
) -> BaseChatModel:
    """Create a ChatAnthropic model from Claude Code gateway conventions."""
    headers = parse_anthropic_custom_headers(custom_headers)
    logger.info(
        "provider.anthropic.create model=%s base_url=%s custom_header_count=%d",
        model,
        base_url,
        len(headers),
    )
    return _AuthTokenChatAnthropic.model_validate(
        {
            "model": model,
            "auth_token": SecretStr(auth_token),
            "base_url": base_url,
            "default_headers": headers,
            "temperature": 0,
        }
    )


def parse_anthropic_custom_headers(value: str) -> Mapping[str, str]:
    """Parse Claude Code's newline-separated ``Name: Value`` header format."""
    headers: dict[str, str] = {}
    for line_number, raw_line in enumerate(value.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        name, separator, header_value = line.partition(":")
        name = name.strip()
        header_value = header_value.strip()
        if not separator or not name or not header_value:
            raise ValueError(
                "ANTHROPIC_CUSTOM_HEADERS line "
                f"{line_number} must use 'Name: Value' format"
            )
        headers[name] = header_value
    return headers
