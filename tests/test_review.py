from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from review_sheep import (
    Confidence,
    CustomLens,
    Finding,
    Lens,
    Location,
    RepositoryMirrorLockTimeout,
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
from review_sheep.backends import ReviewFilesystemBackend


def _checkout(root: Path) -> ReviewCheckout:
    return ReviewCheckout(
        repo="acme/widgets",
        pull_request_number=42,
        base_sha="base123",
        head_sha="head456",
        root=root,
    )


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

    assert "permission denied" in write.func("/review.txt", "changed", runtime).content
    assert (
        "permission denied" in edit.func("/review.txt", "old", "new", runtime).content
    )
    assert "permission denied" in delete.func("/review.txt", runtime).content


class FakeCheckoutSource:
    def __init__(self, checkout: ReviewCheckout) -> None:
        self.checkout = checkout
        self.calls: list[dict[str, Any]] = []

    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout:
        self.calls.append({"repo": repo, "number": number})
        return self.checkout


class DisposableFakeCheckoutSource(FakeCheckoutSource):
    def __init__(self, checkout: ReviewCheckout) -> None:
        super().__init__(checkout)
        self.cleaned: list[ReviewCheckout] = []

    def cleanup_checkout(self, checkout: ReviewCheckout) -> None:
        self.cleaned.append(checkout)


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


class LockTimeoutCheckoutSource:
    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout:
        raise RepositoryMirrorLockTimeout(
            "timed out acquiring repository mirror lock for acme/widgets after 0.050s"
        )


class FailingReviewRunner:
    def run(self, checkout: ReviewCheckout) -> list[Finding]:
        raise TimeoutError("model timed out")


class FakeLensAgent:
    def __init__(
        self,
        finding: dict[str, Any],
        caller_tool: BaseTool | None = None,
        structured_response: Any | None = None,
        on_invoke: Callable[[], Any] | None = None,
    ) -> None:
        self.finding = finding
        self.caller_tool = caller_tool
        self.structured_response = structured_response
        self.on_invoke = on_invoke
        self.calls: list[dict[str, Any]] = []
        self.configs: list[RunnableConfig] = []

    def invoke(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(state)
        self.configs.append(ensure_config())
        if self.on_invoke is not None:
            self.on_invoke()
        if self.caller_tool is not None:
            metadata = self.caller_tool.invoke({"table": "orders"})
            self.finding["description"] = f"Database metadata: {metadata}"
        return {
            "structured_response": self.structured_response
            or {"findings": [self.finding]}
        }


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


def test_review_cleans_a_disposable_checkout_after_success(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    source = DisposableFakeCheckoutSource(checkout)

    result = create_review_agent(
        source=source,
        runner=DeterministicReviewRunner(),
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert source.cleaned == [checkout]


def test_review_cleans_a_disposable_checkout_after_runner_failure(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path)
    source = DisposableFakeCheckoutSource(checkout)

    result = create_review_agent(
        source=source,
        runner=FailingReviewRunner(),
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, ReviewError)
    assert source.cleaned == [checkout]


def test_review_keeps_the_result_when_cleanup_fails(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)

    class FailingCleanupSource(DisposableFakeCheckoutSource):
        def cleanup_checkout(self, checkout: ReviewCheckout) -> None:
            self.cleaned.append(checkout)
            raise OSError("worktree cleanup failed")

    source = FailingCleanupSource(checkout)
    result = create_review_agent(
        source=source,
        runner=DeterministicReviewRunner(),
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert source.cleaned == [checkout]


def test_review_reports_mirror_lock_timeout_as_a_prepare_error() -> None:
    result = create_review_agent(
        source=LockTimeoutCheckoutSource(),
        runner=DeterministicReviewRunner(),
    ).review(repo="acme/widgets", number=42)

    assert result == ReviewError(
        repo="acme/widgets",
        pull_request_number=42,
        operation=ReviewOperation.PREPARE_CHECKOUT,
        error_type="RepositoryMirrorLockTimeout",
        message=(
            "timed out acquiring repository mirror lock for acme/widgets after 0.050s"
        ),
    )


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
    checkout = _checkout(tmp_path).model_copy(
        update={"title": "Guard the caller", "body": "Validates input first."}
    )
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
        lens_concurrency=1,
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert len(factory_calls) == 3
    assert {call["response_format"].schema for call in factory_calls} == {
        lenses_module.CorrectnessResult,
        lenses_module.SecurityResult,
        lenses_module.ConventionsAndTestsResult,
    }
    assert all(
        isinstance(call["response_format"], ToolStrategy) for call in factory_calls
    )
    assert all(call["model"] == "openai:gpt-5-mini" for call in factory_calls)
    assert all(
        isinstance(call["backend"], ReviewFilesystemBackend) for call in factory_calls
    )
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
        "repository-root broad glob" in call["system_prompt"] for call in factory_calls
    )
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
    assert all(
        "<pr_description>\nTitle: Guard the caller\n\nValidates input first.\n"
        "</pr_description>" in agent.calls[0]["messages"][0]["content"]
        for agent in created_agents
    )
    assert [finding.lens for finding in result.findings] == list(Lens)


def test_deep_review_runs_custom_lenses_after_built_ins_in_declaration_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _checkout(tmp_path)
    source = FakeCheckoutSource(checkout)
    custom_lenses = [
        CustomLens(
            identifier="api-compatibility",
            instructions="Review public API changes for compatibility regressions.",
        ),
        CustomLens(
            identifier="migration-safety",
            instructions="Review database migrations for safe rollout behavior.",
        ),
    ]
    factory_calls: list[dict[str, Any]] = []
    agents: list[FakeLensAgent] = []
    metadata_requests: list[str] = []
    start_barrier = threading.Barrier(len(Lens) + len(custom_lenses), timeout=2)

    @tool
    def fetch_db_metadata(table: str) -> str:
        """Fetch read-only metadata for one database table."""
        metadata_requests.append(table)
        return "orders(id, total)"

    def fake_create_deep_agent(**kwargs: Any) -> FakeLensAgent:
        factory_calls.append(kwargs)
        response_format = kwargs["response_format"]
        built_in_lenses = {
            lenses_module.CorrectnessResult: "correctness",
            lenses_module.SecurityResult: "security",
            lenses_module.ConventionsAndTestsResult: "conventions-and-tests",
        }
        lens = built_in_lenses.get(response_format.schema)
        if lens is None:
            lens = next(
                custom.identifier
                for custom in custom_lenses
                if custom.identifier in kwargs["system_prompt"]
            )
        finding = {
            "description": f"Finding from {lens}.",
            "location": {"path": "src/example.py"},
            "severity": "medium",
            "confidence": "likely",
            "lens": lens,
        }
        structured_response = (
            response_format.schema.model_validate({"findings": [finding]})
            if lens not in built_in_lenses.values()
            else None
        )
        agent = FakeLensAgent(
            finding,
            caller_tool=fetch_db_metadata,
            structured_response=structured_response,
            on_invoke=start_barrier.wait,
        )
        agents.append(agent)
        return agent

    monkeypatch.setattr(review_module, "_create_deep_agent", fake_create_deep_agent)

    result = create_deep_review_agent(
        source=source,
        model="openai:gpt-test",
        custom_lenses=custom_lenses,
        tools=[fetch_db_metadata],
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert [finding.lens for finding in result.findings] == [
        *list(Lens),
        "api-compatibility",
        "migration-safety",
    ]
    assert len(factory_calls) == len(Lens) + len(custom_lenses)
    assert source.calls == [{"repo": "acme/widgets", "number": 42}]
    assert all(call["permissions"][0].operations == ["write"] for call in factory_calls)
    assert all(call["permissions"][0].mode == "deny" for call in factory_calls)
    custom_calls = [
        call
        for call in factory_calls
        if any(custom.identifier in call["system_prompt"] for custom in custom_lenses)
    ]
    assert len(custom_calls) == len(custom_lenses)
    for call in custom_calls:
        _assert_filesystem_mutations_denied(call["backend"], call["permissions"])
    assert {agent.configs[0]["metadata"]["review_sheep_lens"] for agent in agents} == {
        *{lens.value for lens in Lens},
        *(lens.identifier for lens in custom_lenses),
    }
    assert metadata_requests == ["orders"] * (len(Lens) + len(custom_lenses))
    assert all(
        [candidate.name for candidate in call["tools"]]
        == ["list_changed_files", "get_diff", "fetch_db_metadata"]
        for call in factory_calls
    )
    assert all(
        custom.instructions in call["system_prompt"]
        for custom in custom_lenses
        for call in factory_calls
        if custom.identifier in call["system_prompt"]
    )
    assert all(isinstance(call["backend"], FilesystemBackend) for call in factory_calls)


def test_deep_review_rejects_duplicate_custom_lenses_before_checkout(
    tmp_path: Path,
) -> None:
    source = FakeCheckoutSource(_checkout(tmp_path))
    custom_lens = CustomLens(
        identifier="api-compatibility",
        instructions="Review public API changes.",
    )

    with pytest.raises(ValueError, match="duplicate"):
        create_deep_review_agent(
            source=source,
            model="openai:gpt-test",
            custom_lenses=[custom_lens, custom_lens],
        )

    assert source.calls == []


def test_deep_review_reports_failing_custom_lens_identity(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    source = FakeCheckoutSource(_checkout(tmp_path))
    custom_lens = CustomLens(
        identifier="api-compatibility",
        instructions="Review public API compatibility.",
    )
    built_in_lenses = {
        lenses_module.CorrectnessResult: "correctness",
        lenses_module.SecurityResult: "security",
        lenses_module.ConventionsAndTestsResult: "conventions-and-tests",
    }

    class FailingLensAgent:
        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("custom gateway unavailable")

    def fake_create_deep_agent(**kwargs: Any) -> FakeLensAgent | FailingLensAgent:
        schema = kwargs["response_format"].schema
        if schema not in built_in_lenses:
            return FailingLensAgent()
        return FakeLensAgent(
            {
                "description": f"Finding from {built_in_lenses[schema]}.",
                "location": {"path": "src/example.py"},
                "severity": "medium",
                "confidence": "likely",
                "lens": built_in_lenses[schema],
            }
        )

    monkeypatch.setattr(review_module, "_create_deep_agent", fake_create_deep_agent)

    result = create_deep_review_agent(
        source=source,
        model="openai:gpt-test",
        custom_lenses=[custom_lens],
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, ReviewError)
    assert result.lens == "api-compatibility"
    assert "api-compatibility" in result.message
    assert "lenses.lens.failed lens=api-compatibility" in caplog.text


def test_deep_review_exposes_caller_tools_to_every_lens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _checkout(tmp_path)
    source = FakeCheckoutSource(checkout)
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
            lenses_module.CorrectnessResult: "correctness",
            lenses_module.SecurityResult: "security",
            lenses_module.ConventionsAndTestsResult: "conventions-and-tests",
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

    monkeypatch.setattr(review_module, "_create_deep_agent", fake_create_deep_agent)

    result = create_deep_review_agent(
        source=source,
        model="openai:gpt-test",
        tools=[fetch_db_metadata],
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert metadata_requests == ["orders"] * len(Lens)
    assert all(
        [candidate.name for candidate in call["tools"]]
        == ["list_changed_files", "get_diff", "fetch_db_metadata"]
        for call in factory_calls
    )
    assert all(
        "Database metadata: orders(id, total)" in finding.description
        for finding in result.findings
    )


def test_deep_review_propagates_trace_config_to_every_lens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _checkout(tmp_path)
    source = FakeCheckoutSource(checkout)
    agents: list[FakeLensAgent] = []
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
        agent = FakeLensAgent(findings_by_schema[kwargs["response_format"].schema])
        agents.append(agent)
        return agent

    monkeypatch.setattr(review_module, "_create_deep_agent", fake_create_deep_agent)
    config: RunnableConfig = {
        "metadata": {
            "review_sheep_run_kind": "ci",
            "review_sheep_session_id": "ci:acme/widgets#42",
        },
        "tags": ["review-sheep", "ci", "review"],
    }

    result = create_deep_review_agent(
        source=source,
        model="openai:gpt-test",
    ).review(repo="acme/widgets", number=42, config=config)

    assert isinstance(result, Review)
    assert len(agents) == len(Lens)
    assert all(
        agent.configs[0]["metadata"]["review_sheep_run_kind"] == "ci"
        for agent in agents
    )
    assert all(
        agent.configs[0]["metadata"]["review_sheep_session_id"] == "ci:acme/widgets#42"
        for agent in agents
    )
    assert {agent.configs[0]["metadata"]["review_sheep_lens"] for agent in agents} == {
        lens.value for lens in Lens
    }
    assert all(
        agent.configs[0]["tags"][:3] == ["review-sheep", "ci", "review"]
        for agent in agents
    )
    assert all(
        agent.configs[0]["tags"][3].startswith("review-sheep-lens:") for agent in agents
    )


@pytest.mark.parametrize(
    "path",
    ["../secret", "../../secret", "/src/../../secret", r"..\secret"],
)
def test_checkout_diff_tool_rejects_parent_traversal(path: str, tmp_path: Path) -> None:
    tools = {
        tool.name: tool for tool in review_module._checkout_tools(_checkout(tmp_path))
    }

    with pytest.raises(ValueError, match="stay inside"):
        tools["get_diff"].invoke({"path": path})


def test_checkout_tools_report_empty_results_and_normalize_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    changed_paths: list[str] = []

    def no_changed_files(_: ReviewCheckout) -> str:
        return ""

    def no_diff(_: ReviewCheckout, *, path: str = "") -> str:
        changed_paths.append(path)
        return ""

    monkeypatch.setattr(review_module, "git_changed_files", no_changed_files)
    monkeypatch.setattr(review_module, "git_diff", no_diff)
    tools = {
        tool.name: tool for tool in review_module._checkout_tools(_checkout(tmp_path))
    }

    assert tools["list_changed_files"].invoke({}) == "No changed files."
    assert tools["get_diff"].invoke({}) == "No diff for that path."
    assert (
        tools["get_diff"].invoke({"path": "  /src\\example.py  "})
        == "No diff for that path."
    )
    assert changed_paths == ["", "src/example.py"]


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


def test_review_returns_diff_failure_as_explainable_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def failing_diff(_: ReviewCheckout, *, path: str = "") -> str:
        raise RuntimeError(f"diff unavailable for {path or 'all files'}")

    monkeypatch.setattr(review_module, "git_diff", failing_diff)

    class DiffRunner:
        def run(self, checkout: ReviewCheckout) -> list[Finding]:
            review_module.git_diff(checkout)
            return []

    result = create_review_agent(
        source=FakeCheckoutSource(_checkout(tmp_path)),
        runner=DiffRunner(),
    ).review(repo="acme/widgets", number=42)

    assert result == ReviewError(
        repo="acme/widgets",
        pull_request_number=42,
        operation=ReviewOperation.RUN_REVIEW,
        error_type="RuntimeError",
        message="diff unavailable for all files",
    )
