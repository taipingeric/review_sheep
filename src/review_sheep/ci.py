"""Non-interactive pull-request Review entrypoint for CI."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any, TextIO

from langchain_core.runnables.config import ensure_config, set_config_context

from review_sheep.checkout import GitCheckoutSource
from review_sheep.config import CIConfig
from review_sheep.domain import Review, ReviewError
from review_sheep.github import GitHubPullRequestReader
from review_sheep.providers import (
    AnthropicModelFactory,
    anthropic_model,
    github_client,
)
from review_sheep.report import render_report
from review_sheep.review import create_deep_review_agent
from review_sheep.runtime_logging import configure_console_logging
from review_sheep.tracing import (
    ci_review_config,
    create_tracing_handlers,
    flush_tracing,
)

logger = logging.getLogger(__name__)


def _redact_secrets(message: str, config: CIConfig) -> str:
    secrets = [
        config.github_token,
        config.anthropic_auth_token,
    ]
    for header in config.custom_headers.splitlines():
        _, separator, value = header.partition(":")
        secrets.append(value.strip() if separator else header.strip())
    if config.langfuse is not None:
        secrets.append(config.langfuse.secret_key)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


def _safe_error(error: BaseException, config: CIConfig) -> str:
    return f"{type(error).__name__}: {_redact_secrets(str(error), config)}"


def main(
    *,
    config: CIConfig,
    output: TextIO | None = None,
    error: TextIO | None = None,
    github_factory: Callable[[str], Any] = github_client,
    model_factory: AnthropicModelFactory = anthropic_model,
    reviewer_factory: Callable[..., Any] = create_deep_review_agent,
    tracing_factory: Callable[..., list[Any]] = create_tracing_handlers,
    tracing_flush: Callable[..., None] = flush_tracing,
) -> int:
    """Run one Review, print its Markdown Report, and return a CI-safe exit code."""
    configure_logs = error is None
    output = output or sys.stdout
    error = error or sys.stderr
    if configure_logs:
        configure_console_logging(
            stream=error,
            level=config.review_log_level,
        )
    logger.info(
        "ci.config repo=%s pr=%d checkout=%s model_tier=%s model=%s",
        config.repo,
        config.pull_request_number,
        config.checkout,
        config.model_tier,
        config.model,
    )
    try:
        trace_handlers = tracing_factory(
            langfuse=config.langfuse,
            mlflow=config.mlflow,
        )
    except Exception:  # noqa: BLE001 - tracing must not replace the Review
        logger.error("ci.tracing.initialization_failed")
        trace_handlers = []
    trace_config = ci_review_config(
        handlers=trace_handlers,
        repo=config.repo,
        number=config.pull_request_number,
    )

    try:
        client = github_factory(config.github_token)
    except Exception as github_error:  # noqa: BLE001 - concise CI boundary
        print(f"error: {_safe_error(github_error, config)}", file=error)
        return 1

    try:
        github = GitHubPullRequestReader(client=client)
        try:
            model = model_factory(
                model=config.model,
                auth_token=config.anthropic_auth_token,
                base_url=config.base_url,
                custom_headers=config.custom_headers,
            )
        except (RuntimeError, ValueError) as config_error:
            print(f"error: {_safe_error(config_error, config)}", file=error)
            return 2
        checkout = GitCheckoutSource(revisions=github, root=config.checkout)
        reviewer = reviewer_factory(
            source=checkout,
            model=model,
            instructions=config.instructions,
        )
        with set_config_context(trace_config or ensure_config()) as context:
            result: Review | ReviewError = context.run(
                reviewer.review,
                repo=config.repo,
                number=config.pull_request_number,
            )
        if isinstance(result, ReviewError):
            print(
                f"error: {result.operation.value} failed for "
                f"{result.repo}#{result.pull_request_number}: "
                f"{_redact_secrets(result.message, config)}",
                file=error,
            )
            return 1
        print(render_report(result).text, end="", file=output)
        logger.info("ci.review.complete findings=%d", len(result.findings))
        return 0
    except Exception as review_error:  # noqa: BLE001 - concise CI boundary
        print(f"error: {_safe_error(review_error, config)}", file=error)
        return 1
    finally:
        try:
            tracing_flush(
                langfuse_enabled=config.langfuse is not None,
                langfuse_public_key=(
                    config.langfuse.public_key if config.langfuse is not None else None
                ),
                mlflow_enabled=config.mlflow is not None,
            )
        except Exception:  # noqa: BLE001 - tracing must not replace the Review
            logger.error("ci.tracing.flush_failed")
        client.close()
        logger.info("ci.stop github_client_closed=true")
