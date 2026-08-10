from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

from review_sheep import Review, ReviewError, ReviewOperation
from review_sheep.ci import main


class FakeGithubClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeReviewer:
    def __init__(self, result: Review | ReviewError) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def review(self, *, repo: str, number: int) -> Review | ReviewError:
        self.calls.append((repo, number))
        return self.result


def _configure(monkeypatch: Any) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "model-key")


def _empty_review() -> Review:
    return Review(
        repo="acme/widgets",
        pull_request_number=42,
        base_sha="base123",
        head_sha="head456",
        findings=[],
    )


def test_ci_prints_one_report_and_closes_github_client(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _configure(monkeypatch)
    output = StringIO()
    errors = StringIO()
    github = FakeGithubClient()
    reviewer = FakeReviewer(_empty_review())
    reviewer_configs: list[dict[str, Any]] = []

    def reviewer_factory(**kwargs: Any) -> FakeReviewer:
        reviewer_configs.append(kwargs)
        return reviewer

    exit_code = main(
        [
            "--repo",
            "acme/widgets",
            "--pr-number",
            "42",
            "--checkout",
            str(tmp_path),
            "--instructions",
            "Focus on authorization.",
        ],
        output=output,
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: "test:model",
        reviewer_factory=reviewer_factory,
    )

    assert exit_code == 0
    assert (
        output.getvalue()
        == """# Review Report: acme/widgets#42

Head: `head456`
Findings: 0

No Findings.
"""
    )
    assert errors.getvalue() == ""
    assert reviewer.calls == [("acme/widgets", 42)]
    assert reviewer_configs[0]["model"] == "test:model"
    assert reviewer_configs[0]["instructions"] == "Focus on authorization."
    assert github.closed is True


def test_ci_returns_review_error_as_failed_exit_code(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _configure(monkeypatch)
    errors = StringIO()
    github = FakeGithubClient()
    reviewer = FakeReviewer(
        ReviewError(
            repo="acme/widgets",
            pull_request_number=42,
            operation=ReviewOperation.PREPARE_CHECKOUT,
            error_type="RuntimeError",
            message="checkout HEAD does not match pull request head",
        )
    )

    exit_code = main(
        [
            "--repo",
            "acme/widgets",
            "--pr-number",
            "42",
            "--checkout",
            str(tmp_path),
        ],
        output=StringIO(),
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: "test:model",
        reviewer_factory=lambda **_: reviewer,
    )

    assert exit_code == 1
    assert errors.getvalue() == (
        "error: prepare_checkout failed for acme/widgets#42: "
        "checkout HEAD does not match pull request head\n"
    )
    assert github.closed is True


def test_ci_requires_pull_request_context_before_creating_clients(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "GITHUB_REPOSITORY",
        "REVIEW_PR_NUMBER",
        "GITHUB_TOKEN",
        "OPENAI_MODEL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    errors = StringIO()
    github_calls = 0

    def github_factory(_: str) -> FakeGithubClient:
        nonlocal github_calls
        github_calls += 1
        return FakeGithubClient()

    exit_code = main(
        [],
        output=StringIO(),
        error=errors,
        github_factory=github_factory,
    )

    assert exit_code == 2
    assert errors.getvalue() == "error: --repo or GITHUB_REPOSITORY is required\n"
    assert github_calls == 0


def test_reusable_workflow_keeps_tooling_outside_the_review_checkout() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/review-pr.yml"
    ).read_text()

    assert "ref: ${{ github.event.pull_request.head.sha }}" in workflow
    assert "fetch-depth: 0" in workflow
    assert "pull-requests: read" in workflow
    assert "repository: taipingeric/review_sheep" in workflow
    assert "path: review-target" in workflow
    assert "path: review-sheep" in workflow
    assert "python scripts/review_pr.py" in workflow
