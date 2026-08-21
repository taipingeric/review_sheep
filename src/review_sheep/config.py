"""Explicit runtime configuration for Review Sheep entrypoints."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast


def _required(value: str, name: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{name} is required")
    return resolved


@dataclass(frozen=True)
class LangfuseConfig:
    """Configuration for optional Langfuse tracing."""

    public_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    base_url: str = "https://cloud.langfuse.com"
    environment: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_key", _required(self.public_key, "public_key"))
        object.__setattr__(self, "secret_key", _required(self.secret_key, "secret_key"))
        object.__setattr__(self, "base_url", _required(self.base_url, "base_url"))
        object.__setattr__(
            self,
            "environment",
            _required(self.environment, "environment"),
        )

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> LangfuseConfig | None:
        values = os.environ if environ is None else environ
        enabled = values.get("LANGFUSE_TRACING_ENABLED", "").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None

        public_key = values.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = values.get("LANGFUSE_SECRET_KEY", "").strip()
        if not public_key and not secret_key and not enabled:
            return None
        if not public_key or not secret_key:
            raise RuntimeError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must both be configured"
            )
        return cls(
            public_key=public_key,
            secret_key=secret_key,
            base_url=values.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
            environment=values.get("LANGFUSE_TRACING_ENVIRONMENT", "default"),
        )


@dataclass(frozen=True)
class MLflowConfig:
    """Explicit settings for one optional MLflow tracing destination."""

    tracking_uri: str = field(repr=False)
    experiment_name: str = "review-sheep"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tracking_uri",
            _required(self.tracking_uri, "tracking_uri"),
        )
        object.__setattr__(
            self,
            "experiment_name",
            self.experiment_name.strip() or "review-sheep",
        )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> MLflowConfig | None:
        """Build MLflow settings when a tracking destination is configured."""
        values = os.environ if environ is None else environ
        tracking_uri = values.get("MLFLOW_TRACKING_URI", "").strip()
        if not tracking_uri:
            return None
        return cls(
            tracking_uri=tracking_uri,
            experiment_name=values.get("MLFLOW_EXPERIMENT_NAME", "review-sheep"),
        )


@dataclass(frozen=True)
class CIConfig:
    """Explicit runtime configuration for one non-interactive Review."""

    repo: str
    pull_request_number: int
    checkout: str
    instructions: str
    model_tier: Literal["haiku", "sonnet", "opus"]
    model: str
    github_token: str = field(repr=False)
    anthropic_auth_token: str = field(repr=False)
    base_url: str = field(repr=False)
    custom_headers: str = field(default="", repr=False)
    review_log_level: str = "INFO"
    langfuse: LangfuseConfig | None = None
    mlflow: MLflowConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo", _required(self.repo, "repo"))
        if self.pull_request_number <= 0:
            raise ValueError("pull_request_number must be positive")
        object.__setattr__(self, "checkout", _required(self.checkout, "checkout"))
        object.__setattr__(self, "instructions", self.instructions.strip())
        tier = self.model_tier.strip().lower()
        if tier not in {"haiku", "sonnet", "opus"}:
            raise ValueError("model_tier must be haiku, sonnet, or opus")
        object.__setattr__(self, "model_tier", tier)
        object.__setattr__(self, "model", _required(self.model, "model"))
        object.__setattr__(
            self,
            "github_token",
            _required(self.github_token, "github_token"),
        )
        object.__setattr__(
            self,
            "anthropic_auth_token",
            _required(self.anthropic_auth_token, "anthropic_auth_token"),
        )
        object.__setattr__(self, "base_url", _required(self.base_url, "base_url"))
        object.__setattr__(self, "custom_headers", self.custom_headers.strip())
        object.__setattr__(
            self,
            "review_log_level",
            self.review_log_level.strip().upper() or "INFO",
        )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        repo: str | None = None,
        pull_request_number: int | None = None,
        checkout: str | None = None,
        instructions: str | None = None,
        model_tier: str | None = None,
        model: str | None = None,
        github_token: str | None = None,
        anthropic_auth_token: str | None = None,
        base_url: str | None = None,
        custom_headers: str | None = None,
    ) -> CIConfig:
        """Build CI config from environment, with explicit values winning."""
        values = os.environ if environ is None else environ
        resolved_tier = (
            model_tier
            if model_tier is not None
            else values.get("REVIEW_MODEL_TIER", "sonnet")
        )
        resolved_tier = resolved_tier.strip().lower()
        if resolved_tier not in {"haiku", "sonnet", "opus"}:
            raise RuntimeError("REVIEW_MODEL_TIER must be haiku, sonnet, or opus")
        resolved_tier = cast(Literal["haiku", "sonnet", "opus"], resolved_tier)

        raw_pr_number = (
            str(pull_request_number)
            if pull_request_number is not None
            else _environment_required(values, "REVIEW_PR_NUMBER")
        )
        try:
            resolved_pr_number = int(raw_pr_number)
        except ValueError as error:
            raise RuntimeError("REVIEW_PR_NUMBER must be a positive integer") from error
        if resolved_pr_number <= 0:
            raise RuntimeError("REVIEW_PR_NUMBER must be a positive integer")

        model_name = (
            model
            if model is not None
            else _environment_required(
                values,
                f"ANTHROPIC_DEFAULT_{resolved_tier.upper()}_MODEL",
            )
        )
        return cls(
            repo=(
                repo
                if repo is not None
                else _environment_required(values, "GITHUB_REPOSITORY")
            ),
            pull_request_number=resolved_pr_number,
            checkout=(
                checkout
                if checkout is not None
                else values.get("REVIEW_CHECKOUT", values.get("GITHUB_WORKSPACE", "."))
            ),
            instructions=(
                instructions
                if instructions is not None
                else values.get("REVIEW_INSTRUCTIONS", "")
            ),
            model_tier=resolved_tier,
            model=model_name,
            github_token=(
                github_token
                if github_token is not None
                else _environment_required(values, "GITHUB_TOKEN")
            ),
            anthropic_auth_token=(
                anthropic_auth_token
                if anthropic_auth_token is not None
                else _environment_required(values, "ANTHROPIC_AUTH_TOKEN")
            ),
            base_url=(
                base_url
                if base_url is not None
                else _environment_required(values, "ANTHROPIC_BASE_URL")
            ),
            custom_headers=(
                custom_headers
                if custom_headers is not None
                else values.get("ANTHROPIC_CUSTOM_HEADERS", "")
            ),
            review_log_level=values.get("REVIEW_LOG_LEVEL", "INFO"),
            langfuse=LangfuseConfig.from_environment(values),
            mlflow=MLflowConfig.from_environment(values),
        )


@dataclass(frozen=True)
class ChatConfig:
    """All configuration required by the interactive chatbot."""

    github_token: str = field(repr=False)
    model: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    review_log_level: str = "INFO"
    langfuse: LangfuseConfig | None = None
    mlflow: MLflowConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "github_token", _required(self.github_token, "github_token")
        )
        object.__setattr__(self, "model", _required(self.model, "model"))
        object.__setattr__(self, "api_key", _required(self.api_key, "api_key"))
        if self.base_url is not None:
            object.__setattr__(self, "base_url", self.base_url.strip() or None)
        object.__setattr__(
            self,
            "review_log_level",
            self.review_log_level.strip().upper() or "INFO",
        )

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> ChatConfig:
        values = os.environ if environ is None else environ
        return cls(
            github_token=_environment_required(values, "GITHUB_TOKEN"),
            model=_environment_required(values, "OPENAI_MODEL"),
            api_key=_environment_required(values, "OPENAI_API_KEY"),
            base_url=values.get("BASE_URL"),
            review_log_level=values.get("REVIEW_LOG_LEVEL", "INFO"),
            langfuse=LangfuseConfig.from_environment(values),
            mlflow=MLflowConfig.from_environment(values),
        )


def _environment_required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value
