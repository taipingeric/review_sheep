from __future__ import annotations

from pathlib import Path

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
            "REVIEW_LENS_CONCURRENCY": "2",
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
    assert config.lens_concurrency == 2
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


def test_chat_config_uses_safe_defaults_for_optional_values() -> None:
    config = ChatConfig.from_environment(
        {
            "GITHUB_TOKEN": "github-token",
            "OPENAI_MODEL": "gpt-injected",
            "OPENAI_API_KEY": "model-key",
            "BASE_URL": "   ",
            "REVIEW_LOG_LEVEL": "   ",
        }
    )

    assert config.base_url is None
    assert config.review_log_level == "INFO"
    assert config.langfuse is None
    assert config.mlflow is None


@pytest.mark.parametrize(
    "environment",
    [
        {"LANGFUSE_PUBLIC_KEY": "public-key"},
        {"LANGFUSE_SECRET_KEY": "secret-key"},
        {"LANGFUSE_TRACING_ENABLED": "true"},
    ],
)
def test_chat_config_rejects_partial_langfuse_configuration(
    environment: dict[str, str],
) -> None:
    with pytest.raises(
        RuntimeError,
        match="LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must both be configured",
    ):
        ChatConfig.from_environment(
            {
                "GITHUB_TOKEN": "github-token",
                "OPENAI_MODEL": "gpt-injected",
                "OPENAI_API_KEY": "model-key",
                **environment,
            }
        )


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


def test_ci_config_uses_workspace_fallback_and_normalizes_optional_values() -> None:
    config = CIConfig.from_environment(
        {
            "GITHUB_REPOSITORY": "acme/widgets",
            "REVIEW_PR_NUMBER": "42",
            "GITHUB_WORKSPACE": "/github/workspace",
            "GITHUB_TOKEN": "github-token",
            "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "gateway-sonnet",
            "REVIEW_LOG_LEVEL": " debug ",
            "MLFLOW_TRACKING_URI": " http://mlflow ",
            "MLFLOW_EXPERIMENT_NAME": "   ",
        }
    )

    assert config.checkout == "/github/workspace"
    assert config.instructions == ""
    assert config.custom_headers == ""
    assert config.review_log_level == "DEBUG"
    assert config.mlflow == MLflowConfig(tracking_uri="http://mlflow")

    default_checkout = CIConfig.from_environment(
        {
            "GITHUB_REPOSITORY": "acme/widgets",
            "REVIEW_PR_NUMBER": "42",
            "GITHUB_TOKEN": "github-token",
            "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "gateway-sonnet",
        }
    )
    assert default_checkout.checkout == "."


def test_ci_config_reads_repository_cache_controls_from_environment(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "repository-cache"
    config = CIConfig.from_environment(
        {
            "GITHUB_REPOSITORY": "acme/widgets",
            "REVIEW_PR_NUMBER": "42",
            "GITHUB_TOKEN": "github-token",
            "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "gateway-sonnet",
            "REVIEW_CACHE_ROOT": str(cache_root),
            "REVIEW_CACHE_LOCK_TIMEOUT": "4.5",
            "REVIEW_CACHE_WORKTREE_RETENTION": "3600",
            "REVIEW_CACHE_MIRROR_RETENTION": "86400",
            "REVIEW_CACHE_CLEANUP": "true",
            "REVIEW_LENS_CONCURRENCY": "2",
        }
    )

    assert config.cache_root == str(cache_root.resolve())
    assert config.cache_lock_timeout == 4.5
    assert config.cache_worktree_retention == 3600
    assert config.cache_mirror_retention == 86400
    assert config.cache_cleanup is True
    assert config.lens_concurrency == 2


def test_ci_config_explicit_cache_controls_override_environment(
    tmp_path: Path,
) -> None:
    config = CIConfig.from_environment(
        {
            "GITHUB_REPOSITORY": "acme/widgets",
            "REVIEW_PR_NUMBER": "42",
            "GITHUB_TOKEN": "github-token",
            "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "gateway-sonnet",
            "REVIEW_CACHE_ROOT": str(tmp_path / "environment-cache"),
            "REVIEW_CACHE_LOCK_TIMEOUT": "30",
            "REVIEW_CACHE_CLEANUP": "false",
            "REVIEW_LENS_CONCURRENCY": "3",
        },
        cache_root=str(tmp_path / "explicit-cache"),
        cache_lock_timeout=2.0,
        cache_cleanup=True,
        lens_concurrency=1,
    )

    assert config.cache_root == str((tmp_path / "explicit-cache").resolve())
    assert config.cache_lock_timeout == 2.0
    assert config.cache_cleanup is True
    assert config.lens_concurrency == 1


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_ci_config_rejects_invalid_lens_concurrency(value: str) -> None:
    values = {
        "GITHUB_REPOSITORY": "acme/widgets",
        "REVIEW_PR_NUMBER": "42",
        "GITHUB_TOKEN": "github-token",
        "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
        "ANTHROPIC_BASE_URL": "https://gateway.example",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "gateway-sonnet",
        "REVIEW_LENS_CONCURRENCY": value,
    }

    with pytest.raises((RuntimeError, ValueError), match="positive integer"):
        CIConfig.from_environment(values)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"REVIEW_CACHE_LOCK_TIMEOUT": "not-a-number"}, "REVIEW_CACHE_LOCK_TIMEOUT"),
        ({"REVIEW_CACHE_WORKTREE_RETENTION": "-1"}, "non-negative"),
        ({"REVIEW_CACHE_CLEANUP": "sometimes"}, "REVIEW_CACHE_CLEANUP"),
    ],
)
def test_ci_config_rejects_invalid_repository_cache_controls(
    environment: dict[str, str], message: str
) -> None:
    values = {
        "GITHUB_REPOSITORY": "acme/widgets",
        "REVIEW_PR_NUMBER": "42",
        "GITHUB_TOKEN": "github-token",
        "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
        "ANTHROPIC_BASE_URL": "https://gateway.example",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "gateway-sonnet",
        **environment,
    }

    with pytest.raises((RuntimeError, ValueError), match=message):
        CIConfig.from_environment(values)


def test_ci_config_rejects_unsafe_relative_cache_root() -> None:
    with pytest.raises(ValueError, match="absolute"):
        CIConfig(
            repo="acme/widgets",
            pull_request_number=42,
            checkout=".",
            instructions="",
            model_tier="sonnet",
            model="gateway-sonnet",
            github_token="github-token",
            anthropic_auth_token="anthropic-token",
            base_url="https://gateway.example",
            cache_root="relative-cache",
        )


def test_ci_config_rejects_shared_existing_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "shared-cache"
    cache_root.mkdir(mode=0o755)
    cache_root.chmod(0o755)

    with pytest.raises(ValueError, match="permissions 0700"):
        CIConfig(
            repo="acme/widgets",
            pull_request_number=42,
            checkout=".",
            instructions="",
            model_tier="sonnet",
            model="gateway-sonnet",
            github_token="github-token",
            anthropic_auth_token="anthropic-token",
            base_url="https://gateway.example",
            cache_root=str(cache_root),
        )


@pytest.mark.parametrize(
    "field",
    ["cache_lock_timeout", "cache_worktree_retention", "cache_mirror_retention"],
)
def test_ci_config_rejects_non_numeric_cache_values(field: str) -> None:
    values = {
        "repo": "acme/widgets",
        "pull_request_number": 42,
        "checkout": ".",
        "instructions": "",
        "model_tier": "sonnet",
        "model": "gateway-sonnet",
        "github_token": "github-token",
        "anthropic_auth_token": "anthropic-token",
        "base_url": "https://gateway.example",
        field: "not-a-number",
    }

    with pytest.raises((ValueError, TypeError), match="finite non-negative"):
        CIConfig(**values)  # type: ignore[arg-type]


def test_ci_config_rejects_non_boolean_cache_cleanup() -> None:
    with pytest.raises((ValueError, TypeError), match="cache_cleanup"):
        CIConfig(
            repo="acme/widgets",
            pull_request_number=42,
            checkout=".",
            instructions="",
            model_tier="sonnet",
            model="gateway-sonnet",
            github_token="github-token",
            anthropic_auth_token="anthropic-token",
            base_url="https://gateway.example",
            cache_cleanup="yes",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {"REVIEW_MODEL_TIER": "large"},
            "REVIEW_MODEL_TIER must be haiku, sonnet, or opus",
        ),
        (
            {"REVIEW_PR_NUMBER": "not-a-number"},
            "REVIEW_PR_NUMBER must be a positive integer",
        ),
        (
            {"REVIEW_PR_NUMBER": "0"},
            "REVIEW_PR_NUMBER must be a positive integer",
        ),
    ],
)
def test_ci_config_rejects_invalid_environment_values(
    environment: dict[str, str], message: str
) -> None:
    values = {
        "GITHUB_REPOSITORY": "acme/widgets",
        "REVIEW_PR_NUMBER": "42",
        "GITHUB_TOKEN": "github-token",
        "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
        "ANTHROPIC_BASE_URL": "https://gateway.example",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "gateway-sonnet",
        **environment,
    }

    with pytest.raises(RuntimeError, match=message):
        CIConfig.from_environment(values)


def test_ci_config_rejects_invalid_model_tier_and_non_positive_number_directly() -> (
    None
):
    with pytest.raises(ValueError, match="model_tier must be haiku, sonnet, or opus"):
        CIConfig(
            repo="acme/widgets",
            pull_request_number=42,
            checkout=".",
            instructions="",
            model_tier="invalid",  # type: ignore[arg-type]
            model="gateway-sonnet",
            github_token="github-token",
            anthropic_auth_token="anthropic-token",
            base_url="https://gateway.example",
        )

    with pytest.raises(ValueError, match="pull_request_number must be positive"):
        CIConfig(
            repo="acme/widgets",
            pull_request_number=0,
            checkout=".",
            instructions="",
            model_tier="sonnet",
            model="gateway-sonnet",
            github_token="github-token",
            anthropic_auth_token="anthropic-token",
            base_url="https://gateway.example",
        )
