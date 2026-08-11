"""Non-interactive pull-request Review entrypoint for CI."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any, TextIO

from dotenv import load_dotenv

from review_sheep.checkout import GitCheckoutSource
from review_sheep.domain import Review, ReviewError
from review_sheep.github import GitHubPullRequestReader
from review_sheep.providers import (
    AnthropicModelFactory,
    anthropic_model,
    github_client,
)
from review_sheep.report import render_report
from review_sheep.review import create_deep_review_agent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review one pull request from a fixed local Git checkout."
    )
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument(
        "--pr-number",
        type=int,
        default=os.getenv("REVIEW_PR_NUMBER") or None,
    )
    parser.add_argument(
        "--checkout",
        default=os.getenv("REVIEW_CHECKOUT", os.getenv("GITHUB_WORKSPACE", ".")),
    )
    parser.add_argument(
        "--instructions",
        default=os.getenv("REVIEW_INSTRUCTIONS", ""),
    )
    parser.add_argument(
        "--model-tier",
        choices=("haiku", "sonnet", "opus"),
        default=(os.getenv("REVIEW_MODEL_TIER", "").strip() or "sonnet").lower(),
    )
    return parser


def _required(value: str, name: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise RuntimeError(f"{name} is required")
    return resolved


def main(
    argv: Sequence[str] | None = None,
    *,
    output: TextIO | None = None,
    error: TextIO | None = None,
    github_factory: Callable[[str], Any] = github_client,
    model_factory: AnthropicModelFactory = anthropic_model,
    reviewer_factory: Callable[..., Any] = create_deep_review_agent,
) -> int:
    """Run one Review, print its Markdown Report, and return a CI-safe exit code."""
    output = output or sys.stdout
    error = error or sys.stderr
    load_dotenv(dotenv_path=".env")
    args = _parser().parse_args(argv)

    try:
        repo = _required(args.repo, "--repo or GITHUB_REPOSITORY")
        if not isinstance(args.pr_number, int) or args.pr_number <= 0:
            raise RuntimeError("--pr-number or REVIEW_PR_NUMBER must be positive")
        checkout_root = _required(args.checkout, "--checkout or REVIEW_CHECKOUT")
        github_token = _required(os.getenv("GITHUB_TOKEN", ""), "GITHUB_TOKEN")
        model_env = f"ANTHROPIC_DEFAULT_{args.model_tier.upper()}_MODEL"
        model_name = _required(os.getenv(model_env, ""), model_env)
        auth_token = _required(
            os.getenv("ANTHROPIC_AUTH_TOKEN", ""), "ANTHROPIC_AUTH_TOKEN"
        )
        base_url = _required(os.getenv("ANTHROPIC_BASE_URL", ""), "ANTHROPIC_BASE_URL")
        custom_headers = os.getenv("ANTHROPIC_CUSTOM_HEADERS", "")
    except RuntimeError as config_error:
        print(f"error: {config_error}", file=error)
        return 2

    try:
        client = github_factory(github_token)
    except Exception as github_error:  # noqa: BLE001 - concise CI boundary
        print(f"error: {type(github_error).__name__}: {github_error}", file=error)
        return 1

    try:
        github = GitHubPullRequestReader(client=client)
        try:
            model = model_factory(
                model=model_name,
                auth_token=auth_token,
                base_url=base_url,
                custom_headers=custom_headers,
            )
        except (RuntimeError, ValueError) as config_error:
            print(f"error: {config_error}", file=error)
            return 2
        checkout = GitCheckoutSource(revisions=github, root=checkout_root)
        reviewer = reviewer_factory(
            source=checkout,
            model=model,
            instructions=args.instructions,
        )
        result: Review | ReviewError = reviewer.review(
            repo=repo,
            number=args.pr_number,
        )
        if isinstance(result, ReviewError):
            print(
                f"error: {result.operation.value} failed for "
                f"{result.repo}#{result.pull_request_number}: {result.message}",
                file=error,
            )
            return 1
        print(render_report(result).text, end="", file=output)
        return 0
    except Exception as review_error:  # noqa: BLE001 - concise CI boundary
        print(f"error: {type(review_error).__name__}: {review_error}", file=error)
        return 1
    finally:
        client.close()
