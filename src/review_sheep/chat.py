"""Interactive while-loop for repeatedly reviewing one live pull request."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any, Protocol, TextIO

from dotenv import load_dotenv
from github import Auth, Github
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from review_sheep.domain import Review
from review_sheep.github import GitHubPullRequestReader
from review_sheep.report import render_report
from review_sheep.review import create_deep_review_agent


class ModelFactory(Protocol):
    def __call__(
        self, *, model: str, api_key: str, base_url: str | None
    ) -> str | BaseChatModel: ...


def _github_client(token: str) -> Github:
    return Github(auth=Auth.Token(token))


def _openai_model(
    *, model: str, api_key: str, base_url: str | None
) -> BaseChatModel:
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


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


def main(
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO | None = None,
    error: TextIO | None = None,
    github_factory: Callable[[str], Any] = _github_client,
    model_factory: ModelFactory = _openai_model,
) -> int:
    """Read .env and repeatedly Review one selected open pull request."""
    output = output or sys.stdout
    error = error or sys.stderr
    load_dotenv(dotenv_path=".env")

    try:
        token = _required_env("GITHUB_TOKEN")
        model_name = _required_env("OPENAI_MODEL")
        api_key = _required_env("OPENAI_API_KEY")
        model = model_factory(
            model=model_name,
            api_key=api_key,
            base_url=os.getenv("BASE_URL") or None,
        )
        default_repo = os.getenv("GITHUB_REPO", "").strip()
        repo = input_fn(f"Repository [{default_repo}]: ").strip() or default_repo
        if not repo:
            raise RuntimeError("repository is required")
        number_text = input_fn("Open PR number: ").strip()
        number = int(number_text)
        if number < 1:
            raise ValueError("pull-request number must be positive")
    except (RuntimeError, ValueError) as config_error:
        print(f"error: {config_error}", file=error)
        return 2

    client = github_factory(token)
    try:
        github = GitHubPullRequestReader(client=client)
        details = github.get_pull_request(repo=repo, number=number)
        if details["state"] != "open":
            print(f"error: {repo}#{number} is not open", file=error)
            return 2

        print(
            f"Reviewing {repo}#{number}: {details['title']} — {details['url']}",
            file=output,
        )
        print("Enter a Review prompt; type exit or quit to stop.", file=output)

        while True:
            try:
                instructions = input_fn("You: ").strip()
            except EOFError:
                break
            if instructions.lower() in {"exit", "quit"}:
                break
            if not instructions:
                continue

            try:
                reviewer = create_deep_review_agent(
                    source=github,
                    model=model,
                    instructions=instructions,
                )
                result = reviewer.review(repo=repo, number=number)
                if not isinstance(result, Review):
                    print(
                        f"error: {result.operation.value}: {result.error_type}: "
                        f"{result.message}",
                        file=error,
                    )
                    continue
                print(render_report(result).text, file=output, end="")
            except Exception as review_error:  # noqa: BLE001 - keep the loop alive
                print(
                    f"error: {type(review_error).__name__}: {review_error}",
                    file=error,
                )
        return 0
    except Exception as github_error:  # noqa: BLE001 - concise CLI boundary
        print(
            f"error: {type(github_error).__name__}: {github_error}",
            file=error,
        )
        return 1
    finally:
        client.close()
