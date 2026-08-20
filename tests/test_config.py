from __future__ import annotations

import pytest

from review_sheep.config import ChatConfig, LangfuseConfig


def test_chat_config_reads_external_environment() -> None:
    config = ChatConfig.from_environment(
        {
            "GITHUB_TOKEN": "github-token",
            "OPENAI_MODEL": "gpt-injected",
            "OPENAI_API_KEY": "model-key",
            "BASE_URL": " https://model.example ",
            "REVIEW_LOG_LEVEL": "debug",
            "LANGFUSE_PUBLIC_KEY": "public-key",
            "LANGFUSE_SECRET_KEY": "secret-key",
            "LANGFUSE_BASE_URL": "https://trace.example",
            "LANGFUSE_TRACING_ENVIRONMENT": "ci",
        }
    )

    assert config.model == "gpt-injected"
    assert config.base_url == "https://model.example"
    assert config.review_log_level == "DEBUG"
    assert config.langfuse == LangfuseConfig(
        public_key="public-key",
        secret_key="secret-key",
        base_url="https://trace.example",
        environment="ci",
    )
    assert "github-token" not in repr(config)
    assert "model-key" not in repr(config)
    assert "secret-key" not in repr(config)


def test_chat_config_disables_optional_tracing_from_environment() -> None:
    config = ChatConfig.from_environment(
        {
            "GITHUB_TOKEN": "github-token",
            "OPENAI_MODEL": "gpt-injected",
            "OPENAI_API_KEY": "model-key",
            "LANGFUSE_TRACING_ENABLED": "false",
            "LANGFUSE_PUBLIC_KEY": "public-key",
        }
    )

    assert config.langfuse is None


def test_chat_config_reports_missing_environment_value_without_echoing_secret() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not configured"):
        ChatConfig.from_environment(
            {
                "GITHUB_TOKEN": "github-token",
                "OPENAI_MODEL": "gpt-injected",
            }
        )
