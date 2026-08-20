from __future__ import annotations

import pytest

from review_sheep.config import ChatConfig, LangfuseConfig, MLflowConfig


def test_chat_config_reads_external_environment() -> None:
    config = ChatConfig.from_environment(
        {
            "GITHUB_TOKEN": "github-token",
            "OPENAI_MODEL": "gpt-test",
            "OPENAI_API_KEY": "openai-key",
            "BASE_URL": " https://provider.example.com/v1 ",
            "REVIEW_LOG_LEVEL": " DEBUG ",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
            "LANGFUSE_BASE_URL": "https://langfuse.example.com",
            "LANGFUSE_TRACING_ENVIRONMENT": "test",
            "MLFLOW_TRACKING_URI": "https://mlflow.example.com",
            "MLFLOW_EXPERIMENT_NAME": "review-sheep-test",
        }
    )

    assert config.model == "gpt-test"
    assert config.base_url == "https://provider.example.com/v1"
    assert config.review_log_level == "DEBUG"
    assert config.langfuse == LangfuseConfig(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="https://langfuse.example.com",
        environment="test",
    )
    assert config.mlflow == MLflowConfig(
        tracking_uri="https://mlflow.example.com",
        experiment_name="review-sheep-test",
    )
    assert "github-token" not in repr(config)
    assert "openai-key" not in repr(config)
    assert "sk-test" not in repr(config)


def test_chat_config_disables_optional_tracing_from_environment() -> None:
    config = ChatConfig.from_environment(
        {
            "GITHUB_TOKEN": "github-token",
            "OPENAI_MODEL": "gpt-test",
            "OPENAI_API_KEY": "openai-key",
            "LANGFUSE_TRACING_ENABLED": "false",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }
    )

    assert config.langfuse is None
    assert config.mlflow is None


def test_chat_config_disables_mlflow_when_tracking_uri_is_absent() -> None:
    config = ChatConfig.from_environment(
        {
            "GITHUB_TOKEN": "github-token",
            "OPENAI_MODEL": "gpt-test",
            "OPENAI_API_KEY": "openai-key",
            "MLFLOW_EXPERIMENT_NAME": "review-sheep-test",
        }
    )

    assert config.mlflow is None


def test_chat_config_reports_missing_environment_value_without_echoing_secret() -> None:
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN is required") as error:
        ChatConfig.from_environment(
            {
                "OPENAI_MODEL": "gpt-test",
                "OPENAI_API_KEY": "openai-key",
            }
        )

    assert "openai-key" not in str(error.value)
