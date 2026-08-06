from __future__ import annotations

import sys
from io import StringIO
from types import SimpleNamespace
from typing import Any

from langchain.agents.structured_output import ToolStrategy

from review_sheep import review as review_module
from review_sheep.chat import _openai_model, main


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


class FakeLensSubagent:
    def __init__(self, finding: dict[str, Any]) -> None:
        self.finding = finding

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"structured_response": {"findings": [self.finding]}}


def test_openai_model_uses_responses_api(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_chat_openai(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "model"

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=fake_chat_openai),
    )

    model = _openai_model(
        model="claude-sonnet-4-6",
        api_key="test-key",
        base_url="https://models.example.test/v1",
    )

    assert model == "model"
    assert captured["use_responses_api"] is True


def test_chat_reads_env_and_reviews_repeated_prompts_until_quit(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("GITHUB_REPO", "acme/widgets")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://models.example.test/v1")
    github = FakeGithubClient()
    prompts = iter(
        [
            "",
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
