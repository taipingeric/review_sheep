from __future__ import annotations

from typing import Any

import pytest
from deepagents.backends import StateBackend
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config

from review_sheep import (
    Finding,
    Lens,
    ManifestReviewAgent,
    PullRequestSnapshot,
    Review,
    ReviewError,
    ReviewOperation,
    ReviewWorkspace,
    SnapshotFile,
    create_manifest_review_agent,
)
from review_sheep import manifest as manifest_module


class FakeSnapshotSource:
    def __init__(self, snapshot: PullRequestSnapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[tuple[str, int]] = []

    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        self.calls.append((repo, number))
        return self.snapshot


class FailingSnapshotSource:
    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        raise RuntimeError("GitHub is unavailable")


class FakeLensAgent:
    def __init__(self, finding: dict[str, Any]) -> None:
        self.finding = finding
        self.calls: list[dict[str, Any]] = []
        self.configs: list[RunnableConfig] = []

    def invoke(
        self,
        state: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        self.calls.append(state)
        self.configs.append(config or ensure_config())
        return {"structured_response": {"findings": [self.finding]}}


def _fake_lens_agent(schema: Any, agents: list[FakeLensAgent]) -> FakeLensAgent:
    lens = {
        manifest_module.CorrectnessResult: "correctness",
        manifest_module.SecurityResult: "security",
        manifest_module.ConventionsAndTestsResult: "conventions-and-tests",
    }[schema]
    agent = FakeLensAgent(
        {
            "description": f"Finding from {lens}.",
            "location": {"path": "src/example.py"},
            "severity": "medium",
            "confidence": "likely",
            "lens": lens,
        }
    )
    agents.append(agent)
    return agent


class DynamicSnapshotSource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        self.calls.append((repo, number))
        return PullRequestSnapshot(
            repo=repo,
            number=number,
            base_sha=f"base-{repo}-{number}",
            head_sha=f"head-{repo}-{number}",
            files=[],
        )


class RecordingManifestRunner:
    def __init__(self) -> None:
        self.workspaces: list[ReviewWorkspace] = []

    def run(self, workspace: ReviewWorkspace) -> list[Finding]:
        self.workspaces.append(workspace)
        return []


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


def test_manifest_review_runs_every_lens_over_the_same_virtual_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSnapshotSource(_snapshot())
    factory_calls: list[dict[str, Any]] = []
    agents: list[FakeLensAgent] = []

    def fake_create_deep_agent(**kwargs: Any) -> FakeLensAgent:
        factory_calls.append(kwargs)
        return _fake_lens_agent(kwargs["response_format"].schema, agents)

    monkeypatch.setattr(manifest_module, "_create_deep_agent", fake_create_deep_agent)

    result = create_manifest_review_agent(
        source=source,
        model="openai:gpt-test",
        instructions="Focus on regressions.",
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert result.base_sha == "base123"
    assert result.head_sha == "head456"
    assert source.calls == [("acme/widgets", 42)]
    assert [finding.lens for finding in result.findings] == list(Lens)
    assert len(factory_calls) == 3
    assert all(isinstance(call["backend"], StateBackend) for call in factory_calls)
    assert all(
        isinstance(call["response_format"], ToolStrategy) for call in factory_calls
    )
    assert all(
        "call them in parallel in the same assistant message" in call["system_prompt"]
        for call in factory_calls
    )
    snapshots = []
    for agent in agents:
        state = agent.calls[0]
        assert set(state["files"]) == {
            "/manifest.json",
            "/diffs/src/example.py.diff",
        }
        snapshots.append(
            {path: data["content"] for path, data in state["files"].items()}
        )
        assert "Focus on regressions." in state["messages"][0]["content"]
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert '"base_sha": "base123"' in snapshots[0]["/manifest.json"]


def test_manifest_review_propagates_trace_config_to_every_lens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSnapshotSource(_snapshot())
    agents: list[FakeLensAgent] = []

    def fake_create_deep_agent(**kwargs: Any) -> FakeLensAgent:
        return _fake_lens_agent(kwargs["response_format"].schema, agents)

    monkeypatch.setattr(manifest_module, "_create_deep_agent", fake_create_deep_agent)
    config: RunnableConfig = {
        "metadata": {"review_sheep_turn": "turn-17"},
        "tags": ["chat"],
    }

    result = create_manifest_review_agent(
        source=source,
        model="openai:gpt-test",
    ).review(repo="acme/widgets", number=42, config=config)

    assert isinstance(result, Review)
    assert len(agents) == len(Lens)
    assert all(
        agent.configs[0]["metadata"]["review_sheep_turn"] == "turn-17"
        for agent in agents
    )
    assert all(agent.configs[0]["tags"] == ["chat"] for agent in agents)


def test_manifest_review_returns_snapshot_failures_as_data() -> None:
    result = create_manifest_review_agent(
        source=FailingSnapshotSource(),
        model="openai:gpt-test",
    ).review(repo="acme/widgets", number=42)

    assert result == ReviewError(
        repo="acme/widgets",
        pull_request_number=42,
        operation=ReviewOperation.FETCH_SNAPSHOT,
        error_type="RuntimeError",
        message="GitHub is unavailable",
    )


def test_manifest_is_built_dynamically_for_each_requested_repository() -> None:
    source = DynamicSnapshotSource()
    runner = RecordingManifestRunner()
    reviewer = ManifestReviewAgent(source=source, runner=runner)

    first = reviewer.review(repo="langchain-ai/langgraph", number=8569)
    second = reviewer.review(repo="acme/widgets", number=42)

    assert isinstance(first, Review)
    assert isinstance(second, Review)
    assert source.calls == [
        ("langchain-ai/langgraph", 8569),
        ("acme/widgets", 42),
    ]
    assert [workspace.manifest.repo for workspace in runner.workspaces] == [
        "langchain-ai/langgraph",
        "acme/widgets",
    ]
    assert [
        workspace.manifest.pull_request_number for workspace in runner.workspaces
    ] == [
        8569,
        42,
    ]
    assert runner.workspaces[0].manifest is not runner.workspaces[1].manifest
