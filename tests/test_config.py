from __future__ import annotations

import pytest

from review_sheep.config import ChatConfig, CIConfig, LangfuseConfig, MLflowConfig


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
            "MLFLOW_TRACKING_URI": "https://mlflow.example",
            "MLFLOW_EXPERIMENT_NAME": "review-sheep-test",
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
    assert config.mlflow == MLflowConfig(
        tracking_uri="https://mlflow.example",
        experiment_name="review-sheep-test",
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
            "MLFLOW_EXPERIMENT_NAME": "review-sheep-test",
        }
    )

    assert config.langfuse is None
    assert config.mlflow is None


def test_chat_config_reports_missing_environment_value_without_echoing_secret() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not configured") as error:
        ChatConfig.from_environment(
            {
                "GITHUB_TOKEN": "github-token",
                "OPENAI_MODEL": "gpt-injected",
            }
        )

    assert "github-token" not in str(error.value)


def test_ci_config_accepts_explicit_values_over_environment() -> None:
    config = CIConfig.from_environment(
        {
            "GITHUB_REPOSITORY": "environment/repo",
            "REVIEW_PR_NUMBER": "12",
            "REVIEW_CHECKOUT": "/environment/checkout",
            "REVIEW_INSTRUCTIONS": "environment instructions",
            "REVIEW_MODEL_TIER": "haiku",
            "GITHUB_TOKEN": "environment-github-token",
            "ANTHROPIC_AUTH_TOKEN": "environment-anthropic-token",
            "ANTHROPIC_BASE_URL": "https://environment.example",
            "ANTHROPIC_CUSTOM_HEADERS": "X-Environment: true",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "environment-haiku",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "environment-opus",
        },
        repo="explicit/repo",
        pull_request_number=42,
        checkout="/explicit/checkout",
        instructions="explicit instructions",
        model_tier="opus",
        model="explicit-model",
        github_token="explicit-github-token",
        anthropic_auth_token="explicit-anthropic-token",
        base_url="https://explicit.example",
        custom_headers="X-Explicit: true",
    )

    assert config.repo == "explicit/repo"
    assert config.pull_request_number == 42
    assert config.checkout == "/explicit/checkout"
    assert config.instructions == "explicit instructions"
    assert config.model_tier == "opus"
    assert config.model == "explicit-model"
    assert config.github_token == "explicit-github-token"
    assert config.anthropic_auth_token == "explicit-anthropic-token"
    assert config.base_url == "https://explicit.example"
    assert config.custom_headers == "X-Explicit: true"
    assert "explicit-github-token" not in repr(config)
    assert "explicit-anthropic-token" not in repr(config)
    assert "X-Explicit: true" not in repr(config)


def test_ci_config_reads_model_for_selected_tier_and_validates_required_values() -> (
    None
):
    config = CIConfig.from_environment(
        {
            "GITHUB_REPOSITORY": "acme/widgets",
            "REVIEW_PR_NUMBER": "42",
            "REVIEW_CHECKOUT": "/workspace",
            "GITHUB_TOKEN": "github-token",
            "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "gateway-sonnet",
        }
    )

    assert config.model == "gateway-sonnet"
    assert config.model_tier == "sonnet"

    with pytest.raises(RuntimeError, match="GITHUB_TOKEN is not configured"):
        CIConfig.from_environment(
            {
                "GITHUB_REPOSITORY": "acme/widgets",
                "REVIEW_PR_NUMBER": "42",
                "REVIEW_CHECKOUT": "/workspace",
                "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
                "ANTHROPIC_BASE_URL": "https://gateway.example",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "gateway-sonnet",
            }
        )
