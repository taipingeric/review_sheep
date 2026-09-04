from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, tool

from review_sheep import (
    CustomLens,
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
    def __init__(
        self,
        finding: dict[str, Any],
        caller_tool: BaseTool | None = None,
    ) -> None:
        self.finding = finding
        self.caller_tool = caller_tool
        self.calls: list[dict[str, Any]] = []
        self.configs: list[RunnableConfig] = []

    def invoke(
        self,
        state: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        self.calls.append(state)
        self.configs.append(config or ensure_config())
        if self.caller_tool is not None:
            metadata = self.caller_tool.invoke({"table": "orders"})
            self.finding["description"] = f"Database metadata: {metadata}"
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


def _assert_filesystem_mutations_denied(backend: Any, permissions: Any) -> None:
    middleware: Any = FilesystemMiddleware(backend=backend, _permissions=permissions)
    runtime: Any = ToolRuntime(
        state={},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="test-mutation",
        store=None,
    )
    write = next(tool for tool in middleware.tools if tool.name == "write_file")
    edit = next(tool for tool in middleware.tools if tool.name == "edit_file")
    delete = next(tool for tool in middleware.tools if tool.name == "delete")

    assert (
        "permission denied" in write.func("/manifest.json", "changed", runtime).content
    )
    assert (
        "permission denied"
        in edit.func("/manifest.json", "old", "new", runtime).content
    )
    assert "permission denied" in delete.func("/manifest.json", runtime).content


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


def test_manifest_review_exposes_caller_tools_to_every_lens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSnapshotSource(_snapshot())
    metadata_requests: list[str] = []
    factory_calls: list[dict[str, Any]] = []

    @tool
    def fetch_db_metadata(table: str) -> str:
        """Fetch read-only metadata for one database table."""
        metadata_requests.append(table)
        return "orders(id, total)"

    def fake_create_deep_agent(**kwargs: Any) -> FakeLensAgent:
        factory_calls.append(kwargs)
        caller_tool = next(
            candidate
            for candidate in kwargs["tools"]
            if candidate.name == "fetch_db_metadata"
        )
        schema = kwargs["response_format"].schema
        lens = {
            manifest_module.CorrectnessResult: "correctness",
            manifest_module.SecurityResult: "security",
            manifest_module.ConventionsAndTestsResult: "conventions-and-tests",
        }[schema]
        return FakeLensAgent(
            {
                "description": "Database metadata not fetched yet.",
                "location": {"path": "src/example.py"},
                "severity": "medium",
                "confidence": "likely",
                "lens": lens,
            },
            caller_tool=caller_tool,
        )

    monkeypatch.setattr(manifest_module, "_create_deep_agent", fake_create_deep_agent)

    result = create_manifest_review_agent(
        source=source,
        model="openai:gpt-test",
        tools=[fetch_db_metadata],
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert metadata_requests == ["orders"] * len(Lens)
    assert all(
        [candidate.name for candidate in call["tools"]] == ["fetch_db_metadata"]
        for call in factory_calls
    )
    assert all(
        "Database metadata: orders(id, total)" in finding.description
        for finding in result.findings
    )


def test_manifest_review_runs_custom_lenses_on_shared_workspace_with_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = FakeSnapshotSource(_snapshot())
    custom_lenses = [
        CustomLens(
            identifier="api-compatibility",
            instructions="Review public API compatibility.",
        ),
        CustomLens(
            identifier="migration-safety",
            instructions="Review database migration safety.",
        ),
    ]
    factory_calls: list[dict[str, Any]] = []
    agents: list[FakeLensAgent] = []
    tool_requests: list[str] = []

    @tool
    def fetch_db_metadata(table: str) -> str:
        """Fetch read-only metadata for one database table."""
        tool_requests.append(table)
        return "orders(id, total)"

    def fake_create_deep_agent(**kwargs: Any) -> FakeLensAgent:
        factory_calls.append(kwargs)
        schema = kwargs["response_format"].schema
        built_in_lenses = {
            manifest_module.CorrectnessResult: "correctness",
            manifest_module.SecurityResult: "security",
            manifest_module.ConventionsAndTestsResult: "conventions-and-tests",
        }
        identity = built_in_lenses.get(schema)
        if identity is None:
            identity = next(
                custom.identifier
                for custom in custom_lenses
                if custom.identifier in kwargs["system_prompt"]
            )
        agent = FakeLensAgent(
            {
                "description": f"Finding from {identity}.",
                "location": {"path": "src/example.py"},
                "severity": "medium",
                "confidence": "likely",
                "lens": identity,
            },
            caller_tool=next(
                candidate
                for candidate in kwargs["tools"]
                if candidate.name == "fetch_db_metadata"
            ),
        )
        agents.append(agent)
        return agent

    monkeypatch.setattr(manifest_module, "_create_deep_agent", fake_create_deep_agent)

    result = create_manifest_review_agent(
        source=source,
        model="openai:gpt-test",
        tools=[fetch_db_metadata],
        custom_lenses=custom_lenses,
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert source.calls == [("acme/widgets", 42)]
    assert [finding.lens for finding in result.findings] == [
        *list(Lens),
        "api-compatibility",
        "migration-safety",
    ]
    assert len(factory_calls) == len(Lens) + len(custom_lenses)
    assert tool_requests == ["orders"] * (len(Lens) + len(custom_lenses))
    assert all(isinstance(call["backend"], StateBackend) for call in factory_calls)
    assert all(call["permissions"][0].operations == ["write"] for call in factory_calls)
    assert all(call["permissions"][0].mode == "deny" for call in factory_calls)
    _assert_filesystem_mutations_denied(
        FilesystemBackend(root_dir=tmp_path), factory_calls[-1]["permissions"]
    )
    assert {agent.configs[0]["metadata"]["review_sheep_lens"] for agent in agents} == {
        *{lens.value for lens in Lens},
        *(lens.identifier for lens in custom_lenses),
    }
    snapshots = [
        {path: data["content"] for path, data in agent.calls[0]["files"].items()}
        for agent in agents
    ]
    assert snapshots[0] == snapshots[1] == snapshots[2] == snapshots[3] == snapshots[4]
    custom_agents = [
        agent
        for agent in agents
        if "api-compatibility" in agent.calls[0]["messages"][0]["content"]
        or "migration-safety" in agent.calls[0]["messages"][0]["content"]
    ]
    assert len(custom_agents) == len(custom_lenses)
    assert {
        next(
            custom.instructions
            for custom in custom_lenses
            if custom.instructions in call["system_prompt"]
        )
        for call in factory_calls
        if any(custom.identifier in call["system_prompt"] for custom in custom_lenses)
    } == {custom.instructions for custom in custom_lenses}


def test_manifest_review_reports_failing_custom_lens_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSnapshotSource(_snapshot())
    custom_lens = CustomLens(
        identifier="api-compatibility",
        instructions="Review public API compatibility.",
    )
    built_in_lenses = {
        manifest_module.CorrectnessResult: "correctness",
        manifest_module.SecurityResult: "security",
        manifest_module.ConventionsAndTestsResult: "conventions-and-tests",
    }
    agents: list[FakeLensAgent] = []

    class FailingLensAgent:
        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("custom gateway unavailable")

    def fake_create_deep_agent(**kwargs: Any) -> FakeLensAgent | FailingLensAgent:
        schema = kwargs["response_format"].schema
        if schema not in built_in_lenses:
            return FailingLensAgent()
        return _fake_lens_agent(schema, agents)

    monkeypatch.setattr(manifest_module, "_create_deep_agent", fake_create_deep_agent)

    result = create_manifest_review_agent(
        source=source,
        model="openai:gpt-test",
        custom_lenses=[custom_lens],
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, ReviewError)
    assert result.lens == "api-compatibility"
    assert result.error_type == "RuntimeError"
    assert "api-compatibility" in result.message


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
    assert {agent.configs[0]["metadata"]["review_sheep_lens"] for agent in agents} == {
        lens.value for lens in Lens
    }
    assert all(agent.configs[0]["tags"][:1] == ["chat"] for agent in agents)
    assert all(
        agent.configs[0]["tags"][1].startswith("review-sheep-lens:") for agent in agents
    )


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
