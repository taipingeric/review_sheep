from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config

from review_sheep import Review, ReviewError, ReviewOperation
from review_sheep.ci import main
from review_sheep.config import CIConfig, LangfuseConfig, MLflowConfig


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


class TraceRecordingReviewer(FakeReviewer):
    def __init__(self, result: Review | ReviewError) -> None:
        super().__init__(result)
        self.configs: list[RunnableConfig] = []

    def review(self, *, repo: str, number: int) -> Review | ReviewError:
        self.configs.append(ensure_config())
        return super().review(repo=repo, number=number)


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


def _ci_config(
    *,
    checkout: str = "/workspace",
    model_tier: Literal["haiku", "sonnet", "opus"] = "sonnet",
    model: str = "gateway-sonnet",
    langfuse: LangfuseConfig | None = None,
    mlflow: MLflowConfig | None = None,
) -> CIConfig:
    return CIConfig(
        repo="acme/widgets",
        pull_request_number=42,
        checkout=checkout,
        instructions="Focus on authorization.",
        model_tier=model_tier,
        model=model,
        github_token="github-token",
        anthropic_auth_token="auth-token",
        base_url="https://gateway.example.com",
        custom_headers="X-Team: review-sheep",
        langfuse=langfuse,
        mlflow=mlflow,
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
        config=_ci_config(checkout=str(tmp_path)),
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


def test_ci_traces_review_with_both_backends_and_flushes_afterward(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _configure(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-ci")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-ci")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-ci")
    output = StringIO()
    errors = StringIO()
    github = FakeGithubClient()
    reviewer = TraceRecordingReviewer(_empty_review())
    tracing_configs: list[dict[str, Any]] = []
    flush_configs: list[dict[str, Any]] = []

    def capture_tracing(**kwargs: Any) -> list[Any]:
        tracing_configs.append(kwargs)
        return ["trace-handler"]

    def capture_flush(**kwargs: Any) -> None:
        flush_configs.append(kwargs)

    exit_code = main(
        config=_ci_config(
            checkout=str(tmp_path),
            langfuse=LangfuseConfig(public_key="pk-ci", secret_key="sk-ci"),
            mlflow=MLflowConfig(tracking_uri="http://mlflow-ci"),
        ),
        output=output,
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: "test:model",
        reviewer_factory=lambda **_: reviewer,
        tracing_factory=capture_tracing,
        tracing_flush=capture_flush,
    )

    assert exit_code == 0
    assert output.getvalue().startswith("# Review Report: acme/widgets#42")
    assert errors.getvalue() == ""
    assert len(tracing_configs) == 1
    assert tracing_configs[0]["langfuse"].public_key == "pk-ci"
    assert tracing_configs[0]["langfuse"].secret_key == "sk-ci"
    assert tracing_configs[0]["mlflow"].tracking_uri == "http://mlflow-ci"
    assert reviewer.configs[0]["metadata"]["review_sheep_run_kind"] == "ci"
    assert reviewer.configs[0]["metadata"]["review_sheep_session_id"] == (
        "ci:acme/widgets#42"
    )
    assert flush_configs == [
        {
            "langfuse_enabled": True,
            "langfuse_public_key": "pk-ci",
            "mlflow_enabled": True,
        }
    ]
    assert github.closed is True


def test_ci_tracing_initialization_failure_does_not_change_review_result(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _configure(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-ci")
    github = FakeGithubClient()

    def unavailable_tracing(**_: Any) -> list[Any]:
        raise RuntimeError("MLflow unavailable")

    def unavailable_flush(**_: Any) -> None:
        raise RuntimeError("MLflow flush unavailable")

    exit_code = main(
        config=_ci_config(checkout=str(tmp_path)),
        output=StringIO(),
        error=StringIO(),
        github_factory=lambda _: github,
        model_factory=lambda **_: "test:model",
        reviewer_factory=lambda **_: FakeReviewer(_empty_review()),
        tracing_factory=unavailable_tracing,
        tracing_flush=unavailable_flush,
    )

    assert exit_code == 0
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
        config=_ci_config(checkout=str(tmp_path)),
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
    github_calls = 0

    def github_factory(_: str) -> FakeGithubClient:
        nonlocal github_calls
        github_calls += 1
        return FakeGithubClient()

    with pytest.raises(RuntimeError, match="REVIEW_PR_NUMBER is not configured"):
        CIConfig.from_environment({})
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
    assert "ANTHROPIC_AUTH_TOKEN: ${{ secrets.ANTHROPIC_AUTH_TOKEN }}" in workflow
    assert "anthropic_base_url: ${{ vars.ANTHROPIC_BASE_URL }}" in workflow
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
        config=_ci_config(
            checkout=str(tmp_path),
            model_tier="opus",
            model="gateway-opus",
        ),
        output=StringIO(),
        error=StringIO(),
        github_factory=lambda _: FakeGithubClient(),
        model_factory=model_factory,
        reviewer_factory=lambda **_: FakeReviewer(_empty_review()),
    )

    assert exit_code == 0
    assert selected_models == ["gateway-opus"]


def test_ci_uses_explicit_config_over_environment(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "environment-github-token")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "environment-auth-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://environment.example")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "environment-model")
    github_tokens: list[str] = []
    model_configs: list[dict[str, Any]] = []

    def capture_github(token: str) -> FakeGithubClient:
        github_tokens.append(token)
        return FakeGithubClient()

    def capture_model(**kwargs: Any) -> str:
        model_configs.append(kwargs)
        return "test:model"

    exit_code = main(
        config=_ci_config(checkout=str(tmp_path)),
        output=StringIO(),
        error=StringIO(),
        github_factory=capture_github,
        model_factory=capture_model,
        reviewer_factory=lambda **_: FakeReviewer(_empty_review()),
    )

    assert exit_code == 0
    assert github_tokens == ["github-token"]
    assert model_configs == [
        {
            "model": "gateway-sonnet",
            "auth_token": "auth-token",
            "base_url": "https://gateway.example.com",
            "custom_headers": "X-Team: review-sheep",
        }
    ]


def test_ci_redacts_credentials_from_logs_and_error_output(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="review_sheep.ci")
    config = CIConfig(
        repo="acme/widgets",
        pull_request_number=42,
        checkout=str(tmp_path),
        instructions="",
        model_tier="sonnet",
        model="gateway-sonnet",
        github_token="github-secret-injected",
        anthropic_auth_token="anthropic-secret-injected",
        base_url="https://gateway.example.com",
        custom_headers="X-Secret: custom-header-secret",
    )
    errors = StringIO()

    def unavailable_github(token: str) -> Any:
        raise RuntimeError(
            f"token={token} auth={config.anthropic_auth_token} "
            "header=custom-header-secret"
        )

    exit_code = main(
        config=config,
        output=StringIO(),
        error=errors,
        github_factory=unavailable_github,
    )

    assert exit_code == 1
    assert "github-secret-injected" not in errors.getvalue()
    assert "anthropic-secret-injected" not in errors.getvalue()
    assert "custom-header-secret" not in errors.getvalue()
    assert "github-secret-injected" not in caplog.text
    assert "anthropic-secret-injected" not in caplog.text
    assert "custom-header-secret" not in caplog.text
    assert "[redacted]" in errors.getvalue()
