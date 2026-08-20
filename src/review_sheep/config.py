"""Runtime configuration for Review Sheep entrypoints."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field


def _required(value: str, name: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{name} is required")
    return resolved


@dataclass(frozen=True)
class LangfuseConfig:
    """Explicit settings for one optional Langfuse tracing project."""

    public_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    base_url: str = "https://cloud.langfuse.com"
    environment: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_key", _required(self.public_key, "public_key"))
        object.__setattr__(self, "secret_key", _required(self.secret_key, "secret_key"))
        object.__setattr__(
            self,
            "base_url",
            self.base_url.strip() or "https://cloud.langfuse.com",
        )
        object.__setattr__(self, "environment", self.environment.strip() or "default")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> LangfuseConfig | None:
        """Build tracing settings from an external process environment."""
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
            base_url=values.get("LANGFUSE_BASE_URL", "")
            or "https://cloud.langfuse.com",
            environment=values.get("LANGFUSE_TRACING_ENVIRONMENT", "") or "default",
        )


@dataclass(frozen=True)
class ChatConfig:
    """All settings required to run the interactive Inquiry and Review path."""

    github_token: str = field(repr=False)
    model: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    review_log_level: str = "INFO"
    langfuse: LangfuseConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "github_token", _required(self.github_token, "github_token")
        )
        object.__setattr__(self, "model", _required(self.model, "model"))
        object.__setattr__(self, "api_key", _required(self.api_key, "api_key"))
        base_url = self.base_url.strip() if self.base_url else None
        object.__setattr__(self, "base_url", base_url or None)
        object.__setattr__(
            self,
            "review_log_level",
            self.review_log_level.strip() or "INFO",
        )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> ChatConfig:
        """Build a complete ChatConfig from an external process environment."""
        values = os.environ if environ is None else environ
        return cls(
            github_token=_environment_required(values, "GITHUB_TOKEN"),
            model=_environment_required(values, "OPENAI_MODEL"),
            api_key=_environment_required(values, "OPENAI_API_KEY"),
            base_url=values.get("BASE_URL", "").strip() or None,
            review_log_level=values.get("REVIEW_LOG_LEVEL", "INFO"),
            langfuse=LangfuseConfig.from_environment(values),
        )


def _environment_required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
