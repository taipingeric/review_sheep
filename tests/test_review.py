from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.structured_output import ToolStrategy
from pydantic import ValidationError

from review_sheep import (
    Confidence,
    Finding,
    Lens,
    Location,
    Review,
    ReviewCheckout,
    ReviewError,
    ReviewOperation,
    Severity,
    create_deep_review_agent,
    create_review_agent,
)
from review_sheep import lenses as lenses_module
from review_sheep import review as review_module


def _checkout(root: Path) -> ReviewCheckout:
    return ReviewCheckout(
        repo="acme/widgets",
        pull_request_number=42,
        base_sha="base123",
        head_sha="head456",
        root=root,
    )


class FakeCheckoutSource:
    def __init__(self, checkout: ReviewCheckout) -> None:
        self.checkout = checkout
        self.calls: list[dict[str, Any]] = []

    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout:
        self.calls.append({"repo": repo, "number": number})
        return self.checkout


class DeterministicReviewRunner:
    def __init__(self) -> None:
        self.checkouts: list[ReviewCheckout] = []

    def run(self, checkout: ReviewCheckout) -> list[Finding]:
        self.checkouts.append(checkout)
        return [
            Finding(
                description="The changed caller omits a required argument.",
                location=Location(path="src/caller.py", start_line=12, end_line=12),
                severity=Severity.HIGH,
                confidence=Confidence.CONFIRMED,
                lens=Lens.CORRECTNESS,
            )
        ]


class FailingCheckoutSource:
    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout:
        raise RuntimeError("checkout HEAD does not match pull request head")


class FailingReviewRunner:
    def run(self, checkout: ReviewCheckout) -> list[Finding]:
        raise TimeoutError("model timed out")


class FakeLensAgent:
    def __init__(self, finding: dict[str, Any]) -> None:
        self.finding = finding
        self.calls: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(state)
        return {"structured_response": {"findings": [self.finding]}}


def test_review_runs_over_one_fixed_checkout(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    source = FakeCheckoutSource(checkout)
    runner = DeterministicReviewRunner()

    result = create_review_agent(source=source, runner=runner).review(
        repo="acme/widgets", number=42
    )

    assert isinstance(result, Review)
    assert result.repo == "acme/widgets"
    assert result.pull_request_number == 42
    assert result.base_sha == "base123"
    assert result.head_sha == "head456"
    assert len(result.findings) == 1
    assert source.calls == [{"repo": "acme/widgets", "number": 42}]
    assert runner.checkouts == [checkout]


@pytest.mark.parametrize(
    "values",
    [
        {"start_line": 1, "end_line": 1},
        {"path": "src/caller.py", "start_line": 1},
        {"path": "src/caller.py", "start_line": 0, "end_line": 1},
        {"path": "src/caller.py", "start_line": 12, "end_line": 11},
    ],
)
def test_location_rejects_incomplete_or_invalid_line_ranges(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        Location(**values)


def test_deep_review_builds_each_lens_on_the_same_read_only_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _checkout(tmp_path)
    source = FakeCheckoutSource(checkout)
    factory_calls: list[dict[str, Any]] = []
    created_agents: list[FakeLensAgent] = []
    findings_by_schema: dict[Any, dict[str, Any]] = {
        lenses_module.CorrectnessResult: {
            "description": "The caller passes an invalid value.",
            "location": {"path": "src/caller.py", "start_line": 1, "end_line": 1},
            "severity": "high",
            "confidence": "confirmed",
            "lens": "correctness",
        },
        lenses_module.SecurityResult: {
            "description": "The changed value crosses a trust boundary.",
            "location": {"path": "src/caller.py", "start_line": 1, "end_line": 1},
            "severity": "critical",
            "confidence": "likely",
            "lens": "security",
        },
        lenses_module.ConventionsAndTestsResult: {
            "description": "The changed behavior has no regression test.",
            "location": {"path": "src/caller.py"},
            "severity": "medium",
            "confidence": "confirmed",
            "lens": "conventions-and-tests",
        },
    }

    def fake_create_deep_agent(**kwargs: Any) -> FakeLensAgent:
        factory_calls.append(kwargs)
        response_format = kwargs["response_format"]
        schema = response_format.schema
        agent = FakeLensAgent(findings_by_schema[schema])
        created_agents.append(agent)
        return agent

    monkeypatch.setattr(review_module, "_create_deep_agent", fake_create_deep_agent)

    result = create_deep_review_agent(
        source=source,
        model="openai:gpt-5-mini",
        instructions="Focus on authorization regressions.",
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert len(factory_calls) == 3
    assert [call["response_format"].schema for call in factory_calls] == [
        lenses_module.CorrectnessResult,
        lenses_module.SecurityResult,
        lenses_module.ConventionsAndTestsResult,
    ]
    assert all(
        isinstance(call["response_format"], ToolStrategy) for call in factory_calls
    )
    assert all(call["model"] == "openai:gpt-5-mini" for call in factory_calls)
    assert all(isinstance(call["backend"], FilesystemBackend) for call in factory_calls)
    assert all(call["backend"].cwd == tmp_path.resolve() for call in factory_calls)
    assert all(call["backend"].virtual_mode for call in factory_calls)
    assert all(
        [tool.name for tool in call["tools"]] == ["list_changed_files", "get_diff"]
        for call in factory_calls
    )
    assert all(call["permissions"][0].operations == ["write"] for call in factory_calls)
    assert all(call["permissions"][0].mode == "deny" for call in factory_calls)
    assert all("list_changed_files" in call["system_prompt"] for call in factory_calls)
    assert all(
        "call them in parallel in the same assistant message" in call["system_prompt"]
        for call in factory_calls
    )
    assert all(
        "manifest" not in call["system_prompt"].lower() for call in factory_calls
    )
    assert all(set(agent.calls[0]) == {"messages"} for agent in created_agents)
    assert all(
        "Focus on authorization regressions."
        in agent.calls[0]["messages"][0]["content"]
        for agent in created_agents
    )
    assert [finding.lens for finding in result.findings] == list(Lens)


def test_checkout_diff_tool_rejects_parent_traversal(tmp_path: Path) -> None:
    tools = {
        tool.name: tool for tool in review_module._checkout_tools(_checkout(tmp_path))
    }

    with pytest.raises(ValueError, match="stay inside"):
        tools["get_diff"].invoke({"path": "../secret"})


def test_lens_output_schema_rejects_another_lens_identity() -> None:
    with pytest.raises(ValidationError):
        lenses_module.SecurityResult.model_validate(
            {
                "findings": [
                    {
                        "description": "Wrongly tagged security finding.",
                        "location": {},
                        "severity": "high",
                        "confidence": "confirmed",
                        "lens": "correctness",
                    }
                ]
            }
        )


def test_review_returns_checkout_failure_as_explainable_data() -> None:
    result = create_review_agent(
        source=FailingCheckoutSource(),
        runner=DeterministicReviewRunner(),
    ).review(repo="acme/widgets", number=42)

    assert result == ReviewError(
        repo="acme/widgets",
        pull_request_number=42,
        operation=ReviewOperation.PREPARE_CHECKOUT,
        error_type="RuntimeError",
        message="checkout HEAD does not match pull request head",
    )


def test_review_returns_model_failure_as_explainable_data(tmp_path: Path) -> None:
    result = create_review_agent(
        source=FakeCheckoutSource(_checkout(tmp_path)),
        runner=FailingReviewRunner(),
    ).review(repo="acme/widgets", number=42)

    assert result == ReviewError(
        repo="acme/widgets",
        pull_request_number=42,
        operation=ReviewOperation.RUN_REVIEW,
        error_type="TimeoutError",
        message="model timed out",
    )
