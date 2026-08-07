from __future__ import annotations

import sys
from io import StringIO
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from review_sheep import review as review_module
from review_sheep.chat import main


class FakeRepository:
    def __init__(self) -> None:
        self.file_requests = 0
        self.pull = SimpleNamespace(
            number=42,
            title="Keep the flock together",
            state="open",
            user=SimpleNamespace(login="alice"),
            created_at=None,
            updated_at=None,
            html_url="https://github.com/acme/widgets/pull/42",
            body="Protect the public flock invariant.",
            base=SimpleNamespace(ref="main"),
            head=SimpleNamespace(ref="review", sha="abc123"),
            draft=False,
            merged=False,
            mergeable_state="clean",
            changed_files=1,
            additions=1,
            deletions=1,
            labels=[],
            get_files=self.get_files,
        )

    def get_pull(self, number: int) -> Any:
        assert number == 42
        return self.pull

    def get_files(self) -> list[Any]:
        self.file_requests += 1
        return [
            SimpleNamespace(
                filename="src/auth.py",
                status="modified",
                additions=1,
                deletions=1,
                patch="@@ -8 +8 @@\n-authorize(actor)\n+authorize()",
                previous_filename=None,
            )
        ]


class FakeGithubClient:
    def __init__(self) -> None:
        self.repository = FakeRepository()
        self.closed = False

    def get_repo(self, repo: str) -> FakeRepository:
        assert repo == "acme/widgets"
        return self.repository

    def close(self) -> None:
        self.closed = True


class MissingPullRequestRepository(FakeRepository):
    def get_pull(self, number: int) -> Any:
        assert number == 42
        raise RuntimeError("pull request not found")


class UnavailableSnapshotRepository(FakeRepository):
    def __init__(self) -> None:
        super().__init__()
        self.pull_requests = 0

    def get_pull(self, number: int) -> Any:
        assert number == 42
        self.pull_requests += 1
        if self.pull_requests == 1:
            return self.pull
        raise RuntimeError(f"snapshot unavailable {self.pull_requests - 1}")


class UnusedReviewModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "unused-review-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> UnusedReviewModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("model must not run when snapshot fetching fails")


class FakeLensSubagent:
    def __init__(self, finding: dict[str, Any]) -> None:
        self.finding = finding

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"structured_response": {"findings": [self.finding]}}


def test_chat_reports_github_client_startup_failure(monkeypatch: Any) -> None:
    monkeypatch.setenv("GITHUB_REPO", "acme/widgets")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    prompts = iter(["", "42"])
    output = StringIO()
    errors = StringIO()

    def unavailable_github(_: str) -> Any:
        raise RuntimeError("GitHub credentials rejected")

    exit_code = main(
        input_fn=lambda _: next(prompts),
        output=output,
        error=errors,
        github_factory=unavailable_github,
        model_factory=lambda **_: "test-model",
    )

    assert exit_code == 1
    assert output.getvalue() == ""
    assert errors.getvalue() == "error: RuntimeError: GitHub credentials rejected\n"


def test_chat_exits_cleanly_on_eof_during_the_conversation(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    output = StringIO()
    errors = StringIO()
    github = FakeGithubClient()

    def end_input(_: str) -> str:
        raise EOFError

    exit_code = main(
        input_fn=end_input,
        output=output,
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: "test-model",
    )

    assert exit_code == 0
    assert output.getvalue().startswith("Review chatbot ready;")
    assert errors.getvalue() == ""
    assert github.closed is True


def test_chat_treats_a_whitespace_base_url_as_unset(monkeypatch: Any) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "   ")
    base_urls: list[str | None] = []

    def capture_model_config(**kwargs: Any) -> str:
        base_urls.append(kwargs["base_url"])
        return "test-model"

    github = FakeGithubClient()
    prompts = iter(["quit"])

    exit_code = main(
        input_fn=lambda _: next(prompts),
        output=StringIO(),
        error=StringIO(),
        github_factory=lambda _: github,
        model_factory=capture_model_config,
    )

    assert exit_code == 0
    assert base_urls == [None]
    assert github.closed is True


@pytest.mark.parametrize(
    ("availability", "expected_reply"),
    [
        (
            "missing",
            (
                "Bot: I could not read acme/widgets#42: RuntimeError: "
                "pull request not found. Try another number.\n"
            ),
        ),
        (
            "closed",
            "Bot: acme/widgets#42 is not open. Try another number.\n",
        ),
    ],
)
def test_chat_asks_for_another_pull_request_when_one_is_unavailable(
    monkeypatch: Any,
    availability: str,
    expected_reply: str,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    repository = (
        MissingPullRequestRepository()
        if availability == "missing"
        else FakeRepository()
    )
    if availability == "closed":
        repository.pull.state = "closed"
    github = FakeGithubClient()
    github.repository = repository
    output = StringIO()
    errors = StringIO()
    prompts = iter(["acme/widgets", "42", "quit"])

    exit_code = main(
        input_fn=lambda _: next(prompts),
        output=output,
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: "test-model",
    )

    assert exit_code == 0
    assert expected_reply in output.getvalue()
    assert errors.getvalue() == ""
    assert github.closed is True


def test_chat_continues_after_each_review_snapshot_failure(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    github = FakeGithubClient()
    github.repository = UnavailableSnapshotRepository()
    prompts = iter(
        [
            "acme/widgets",
            "42",
            "Review correctness.",
            "Review security.",
            "quit",
        ]
    )
    output = StringIO()
    errors = StringIO()

    exit_code = main(
        input_fn=lambda _: next(prompts),
        output=output,
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: UnusedReviewModel(),
    )

    assert exit_code == 0
    assert output.getvalue().count(
        "Bot: Review failed during fetch_snapshot: RuntimeError: "
        "snapshot unavailable"
    ) == 2
    assert errors.getvalue() == ""
    assert github.closed is True


@pytest.mark.parametrize("number", ["not-a-number", "0", "-1"])
def test_chat_asks_again_for_an_invalid_pull_request_number(
    monkeypatch: Any,
    number: str,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    github = FakeGithubClient()
    output = StringIO()
    errors = StringIO()
    prompts = iter(["acme/widgets", number, "quit"])

    exit_code = main(
        input_fn=lambda _: next(prompts),
        output=output,
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: "test-model",
    )

    assert exit_code == 0
    assert (
        "Bot: That is not a positive pull-request number. Try again.\n"
        in output.getvalue()
    )
    assert errors.getvalue() == ""
    assert github.closed is True


def test_chat_rejects_missing_required_configuration_before_external_clients(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("GITHUB_TOKEN", "OPENAI_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    errors = StringIO()

    exit_code = main(
        input_fn=lambda _: "",
        output=StringIO(),
        error=errors,
        github_factory=lambda _: (_ for _ in ()).throw(
            AssertionError("GitHub client must not be created")
        ),
        model_factory=lambda **_: (_ for _ in ()).throw(
            AssertionError("model must not be created")
        ),
    )

    assert exit_code == 2
    assert errors.getvalue() == "error: GITHUB_TOKEN is not set in .env\n"


def test_chat_configures_the_openai_provider_for_responses_api(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://models.example.test/v1")
    captured: dict[str, Any] = {}

    def fake_chat_openai(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "model"

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=fake_chat_openai),
    )

    github = FakeGithubClient()
    prompts = iter(["quit"])
    exit_code = main(
        input_fn=lambda _: next(prompts),
        output=StringIO(),
        error=StringIO(),
        github_factory=lambda _: github,
    )

    assert exit_code == 0
    assert captured["use_responses_api"] is True
    assert github.closed is True


def test_chat_reads_env_and_reviews_repeated_prompts_until_quit(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://models.example.test/v1")
    github = FakeGithubClient()
    prompts = iter(
        [
            "acme/widgets",
            "42",
            "Focus on correctness regressions.",
            "Now focus on authorization.",
            "quit",
        ]
    )
    output = StringIO()
    errors = StringIO()
    findings: dict[Any, dict[str, Any]] = {
        review_module.CorrectnessResult: {
            "description": "The caller omits the required actor.",
            "location": {"path": "src/auth.py", "start_line": 8, "end_line": 8},
            "severity": "high",
            "confidence": "confirmed",
            "lens": "correctness",
        },
        review_module.SecurityResult: {
            "description": "The changed call bypasses authorization.",
            "location": {"path": "src/auth.py", "start_line": 8, "end_line": 8},
            "severity": "critical",
            "confidence": "confirmed",
            "lens": "security",
        },
        review_module.ConventionsAndTestsResult: {
            "description": "No regression test covers the changed call.",
            "location": {"path": "src/auth.py"},
            "severity": "medium",
            "confidence": "likely",
            "lens": "conventions-and-tests",
        },
    }

    def fake_create_deep_agent(**kwargs: Any) -> FakeLensSubagent:
        response_format = kwargs["response_format"]
        schema = (
            response_format.schema
            if isinstance(response_format, ToolStrategy)
            else response_format
        )
        return FakeLensSubagent(findings[schema])

    monkeypatch.setattr(review_module, "_create_deep_agent", fake_create_deep_agent)

    exit_code = main(
        input_fn=lambda _: next(prompts),
        output=output,
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: "test-model",
    )

    assert exit_code == 0
    assert errors.getvalue() == ""
    assert output.getvalue().count("# Review Report: acme/widgets#42") == 2
    assert github.repository.file_requests == 2
    assert github.closed is True
