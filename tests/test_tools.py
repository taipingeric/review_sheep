from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool

from review_sheep import PullRequestSnapshot, SnapshotFile
from review_sheep import manifest as manifest_module
from review_sheep.tools import review_pull_request_tool


class FakeSnapshotSource:
    def __init__(self, snapshot: PullRequestSnapshot) -> None:
        self.snapshot = snapshot

    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        return self.snapshot


class FailingSnapshotSource:
    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        raise RuntimeError("GitHub is unavailable")


class FakeLensAgent:
    def __init__(self, finding: dict[str, Any]) -> None:
        self.finding = finding

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"structured_response": {"findings": [self.finding]}}


def _snapshot() -> PullRequestSnapshot:
    return PullRequestSnapshot(
        repo="acme/widgets",
        number=42,
        base_sha="base123",
        head_sha="head456",
        files=[
            SnapshotFile(
                path="src/example.py",
                status="modified",
                additions=1,
                deletions=1,
                patch="@@ -1 +1 @@\n-old\n+new",
            )
        ],
    )


def _stub_lens_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    schema_to_lens = {
        manifest_module.CorrectnessResult: "correctness",
        manifest_module.SecurityResult: "security",
        manifest_module.ConventionsAndTestsResult: "conventions-and-tests",
    }

    def fake_create_deep_agent(**kwargs: Any) -> FakeLensAgent:
        lens_name = schema_to_lens[kwargs["response_format"].schema]
        return FakeLensAgent(
            {
                "description": f"Finding from {lens_name}.",
                "location": {"path": "src/example.py"},
                "severity": "medium",
                "confidence": "likely",
                "lens": lens_name,
            }
        )

    monkeypatch.setattr(manifest_module, "_create_deep_agent", fake_create_deep_agent)


def test_review_pull_request_tool_is_a_langchain_tool() -> None:
    tool = review_pull_request_tool("openai:gpt-test", FakeSnapshotSource(_snapshot()))

    assert isinstance(tool, BaseTool)
    assert tool.name == "review_pull_request"


def test_review_pull_request_tool_returns_the_rendered_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_lens_agent(monkeypatch)
    tool = review_pull_request_tool("openai:gpt-test", FakeSnapshotSource(_snapshot()))

    result = tool.invoke({"repo": "acme/widgets", "number": 42})

    assert "# Review Report: acme/widgets#42" in result
    assert "Finding from correctness." in result


def test_review_pull_request_tool_returns_errors_as_text_instead_of_raising() -> None:
    tool = review_pull_request_tool("openai:gpt-test", FailingSnapshotSource())

    result = tool.invoke({"repo": "acme/widgets", "number": 42})

    assert result == (
        "fetch_snapshot failed for acme/widgets#42: GitHub is unavailable"
    )
