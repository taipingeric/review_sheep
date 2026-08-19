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
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Team: review-sheep")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_HAIKU_MODEL", "gateway-haiku")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_OPUS_MODEL", "gateway-opus")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "gateway-sonnet")


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
    model_configs: list[dict[str, Any]] = []

    def reviewer_factory(**kwargs: Any) -> FakeReviewer:
        reviewer_configs.append(kwargs)
        return reviewer

    def model_factory(**kwargs: Any) -> str:
        model_configs.append(kwargs)
        return "test:model"

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
        model_factory=model_factory,
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
    assert model_configs == [
        {
            "model": "gateway-sonnet",
            "auth_token": "auth-token",
            "base_url": "https://gateway.example.com",
            "custom_headers": "X-Team: review-sheep",
        }
    ]
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
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
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
    assert "pull-requests: write" in workflow
    assert "repository: taipingeric/review_sheep" in workflow
    assert "path: review-target" in workflow
    assert "path: review-sheep" in workflow
    assert "ANTHROPIC_AUTH_TOKEN: ${{ secrets.ANTHROPIC_AUTH_TOKEN }}" in workflow
    assert (
        "ANTHROPIC_BASE_URL: "
        "${{ inputs.anthropic_base_url || secrets.ANTHROPIC_BASE_URL }}" in workflow
    )
    assert (
        "ANTHROPIC_DEFAULT_SONNET_MODEL: "
        "${{ inputs.sonnet_model || secrets.ANTHROPIC_DEFAULT_SONNET_MODEL }}"
        in workflow
    )
    assert "python scripts/review_pr.py" in workflow
    assert "<!-- review-sheep-report -->" in workflow
    assert "gh api --method PATCH" in workflow
    assert 'gh pr comment "$REVIEW_PR_NUMBER"' in workflow


def test_repository_ci_calls_the_reusable_review_for_pull_requests() -> None:
    workflow = (Path(__file__).parents[1] / ".github/workflows/review.yml").read_text()

    assert "pull_request:" in workflow
    assert "uses: ./.github/workflows/review-pr.yml" in workflow
    assert "review_sheep_ref: ${{ github.event.pull_request.head.sha }}" in workflow
    assert (
        "ANTHROPIC_AUTH_TOKEN: " "${{ secrets.ANTHROPIC_AUTH_TOKEN }}" in workflow
    )
    assert (
        "anthropic_base_url: " "${{ vars.ANTHROPIC_BASE_URL }}" in workflow
    )
    assert "sonnet_model:" in workflow
    assert "claude-sonnet-4-6" in workflow
    assert "head.repo.full_name == github.repository" in workflow
    assert "pull-requests: write" in workflow


def test_ci_can_select_the_opus_gateway_model(monkeypatch: Any, tmp_path: Path) -> None:
    _configure(monkeypatch)
    selected_models: list[str] = []

    def model_factory(**kwargs: Any) -> str:
        selected_models.append(kwargs["model"])
        return "test:model"

    exit_code = main(
        [
            "--repo",
            "acme/widgets",
            "--pr-number",
            "42",
            "--checkout",
            str(tmp_path),
            "--model-tier",
            "opus",
        ],
        output=StringIO(),
        error=StringIO(),
        github_factory=lambda _: FakeGithubClient(),
        model_factory=model_factory,
        reviewer_factory=lambda **_: FakeReviewer(_empty_review()),
    )

    assert exit_code == 0
    assert selected_models == ["gateway-opus"]
