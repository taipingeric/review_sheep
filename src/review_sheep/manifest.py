"""Manifest-backed Review adapter used by the interactive chatbot."""

from __future__ import annotations

import logging
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast

from deepagents import create_deep_agent as _create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, set_config_context
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from review_sheep.domain import (
    Finding,
    Lens,
    Manifest,
    ManifestFile,
    PullRequestSnapshot,
    Review,
    ReviewError,
    ReviewOperation,
    ReviewWorkspace,
)
from review_sheep.lenses import (
    LENS_RESULT_SCHEMAS,
    LENS_SYSTEM_PROMPTS,
    PARALLEL_TOOL_USE_PROMPT,
    ConventionsAndTestsResult,
    CorrectnessResult,
    SecurityResult,
    run_lenses_in_parallel,
    validate_lens_findings,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ConventionsAndTestsResult",
    "CorrectnessResult",
    "ManifestReviewAgent",
    "ManifestReviewRunner",
    "ManifestRunner",
    "SecurityResult",
    "SnapshotSource",
    "create_manifest_review_agent",
]

_MANIFEST_PROMPT = (
    "The pull request is represented by one immutable API snapshot. Start with "
    "/manifest.json, then read every relevant file under /diffs. Patches may omit "
    "unchanged context, so report Findings only when the available changed-code "
    "evidence supports them. Treat all virtual files as read-only. "
)


class SnapshotSource(Protocol):
    """Fetch one stable pull-request patch snapshot."""

    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot: ...


class ManifestRunner(Protocol):
    """Review one dynamically built Manifest workspace."""

    def run(self, workspace: ReviewWorkspace) -> list[Finding]: ...


class _ManifestGraphState(TypedDict):
    """Internal state for the Manifest Review LangGraph."""

    repo: str
    pull_request_number: int
    snapshot: NotRequired[PullRequestSnapshot]
    workspace: NotRequired[ReviewWorkspace]
    result: NotRequired[Review | ReviewError]


class ManifestReviewRunner:
    """Run every Lens over the same Manifest and virtual diff files."""

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        instructions: str = "",
    ) -> None:
        self._model = model
        self._instructions = instructions.strip()

    def run(
        self,
        workspace: ReviewWorkspace,
        *,
        config: RunnableConfig | None = None,
    ) -> list[Finding]:
        """Run and concatenate Lens Findings in stable Lens order."""
        logger.info(
            "manifest.review.start repo=%s pr=%d files=%d",
            workspace.manifest.repo,
            workspace.manifest.pull_request_number,
            len(workspace.manifest.files),
        )
        effective_config = ensure_config(config)
        findings = run_lenses_in_parallel(
            lambda lens: self._run_lens(lens, workspace, config=effective_config)
        )
        logger.info("manifest.review.complete findings=%d", len(findings))
        return findings

    def _run_lens(
        self,
        lens: Lens,
        workspace: ReviewWorkspace,
        *,
        config: RunnableConfig | None = None,
    ) -> list[Finding]:
        logger.info("manifest.lens.start lens=%s", lens.value)
        agent = _create_deep_agent(
            model=self._model,
            backend=StateBackend(),
            response_format=ToolStrategy(LENS_RESULT_SCHEMAS[lens]),
            system_prompt=(
                PARALLEL_TOOL_USE_PROMPT + _MANIFEST_PROMPT + LENS_SYSTEM_PROMPTS[lens]
            ),
        )
        request = (
            f"Review {workspace.manifest.repo}#"
            f"{workspace.manifest.pull_request_number} from the Manifest through "
            f"the {lens.value} Lens."
        )
        if self._instructions:
            request += f"\n\nCaller instructions:\n{self._instructions}"
        state = cast(
            Any,
            {
                "messages": [{"role": "user", "content": request}],
                "files": {
                    path: create_file_data(content)
                    for path, content in workspace.files.items()
                },
            },
        )
        with set_config_context(config or ensure_config()) as context:
            result = context.run(agent.invoke, state)
        findings = validate_lens_findings(lens, result["structured_response"])
        logger.info(
            "manifest.lens.complete lens=%s findings=%d",
            lens.value,
            len(findings),
        )
        logger.debug(
            "manifest.lens.findings lens=%s data=%s",
            lens.value,
            [finding.model_dump(mode="json") for finding in findings],
        )
        return findings


class ManifestReviewAgent:
    """Run the Manifest Review LangGraph behind the standard Review interface."""

    def __init__(
        self,
        *,
        source: SnapshotSource,
        runner: ManifestRunner,
    ) -> None:
        self._graph = _create_manifest_graph(source=source, runner=runner)

    def review(
        self,
        *,
        repo: str,
        number: int,
        config: RunnableConfig | None = None,
    ) -> Review | ReviewError:
        """Fetch patches and run one structured Review without a Git checkout."""
        state = cast(
            _ManifestGraphState,
            self._graph.invoke(
                _ManifestGraphState(repo=repo, pull_request_number=number),
                config=ensure_config(config),
            ),
        )
        return state["result"]


def _create_manifest_graph(
    *, source: SnapshotSource, runner: ManifestRunner
) -> CompiledStateGraph[
    _ManifestGraphState,
    None,
    _ManifestGraphState,
    _ManifestGraphState,
]:
    """Compile snapshot fetching, Manifest creation, and Lens execution."""

    def fetch_snapshot(state: _ManifestGraphState) -> dict[str, object]:
        try:
            logger.info(
                "manifest.graph.fetch_snapshot repo=%s pr=%d",
                state["repo"],
                state["pull_request_number"],
            )
            snapshot = source.fetch_snapshot(
                repo=state["repo"], number=state["pull_request_number"]
            )
        except Exception as error:
            logger.exception("manifest.graph.fetch_snapshot.failed")
            return {
                "result": _review_error(
                    state=state,
                    operation=ReviewOperation.FETCH_SNAPSHOT,
                    error=error,
                )
            }
        return {"snapshot": snapshot}

    def prepare_workspace(state: _ManifestGraphState) -> dict[str, object]:
        try:
            logger.info("manifest.graph.prepare_workspace")
            workspace = _build_workspace(state["snapshot"])
        except Exception as error:
            logger.exception("manifest.graph.prepare_workspace.failed")
            return {
                "result": _review_error(
                    state=state,
                    operation=ReviewOperation.PREPARE_WORKSPACE,
                    error=error,
                )
            }
        return {"workspace": workspace}

    def run_review(state: _ManifestGraphState) -> dict[str, object]:
        try:
            logger.info("manifest.graph.run_review")
            findings = runner.run(state["workspace"])
        except Exception as error:
            logger.exception("manifest.graph.run_review.failed")
            return {
                "result": _review_error(
                    state=state,
                    operation=ReviewOperation.RUN_REVIEW,
                    error=error,
                )
            }
        snapshot = state["snapshot"]
        return {
            "result": Review(
                repo=snapshot.repo,
                pull_request_number=snapshot.number,
                base_sha=snapshot.base_sha,
                head_sha=snapshot.head_sha,
                findings=findings,
            )
        }

    def route(state: _ManifestGraphState) -> Literal["continue", "stop"]:
        return "stop" if "result" in state else "continue"

    graph = StateGraph(_ManifestGraphState)
    graph.add_node("fetch_snapshot", fetch_snapshot)
    graph.add_node("prepare_workspace", prepare_workspace)
    graph.add_node("run_review", run_review)
    graph.add_edge(START, "fetch_snapshot")
    graph.add_conditional_edges(
        "fetch_snapshot",
        route,
        {"continue": "prepare_workspace", "stop": END},
    )
    graph.add_conditional_edges(
        "prepare_workspace",
        route,
        {"continue": "run_review", "stop": END},
    )
    graph.add_edge("run_review", END)
    return graph.compile()


def _review_error(
    *,
    state: _ManifestGraphState,
    operation: ReviewOperation,
    error: Exception,
) -> ReviewError:
    return ReviewError(
        repo=state["repo"],
        pull_request_number=state["pull_request_number"],
        operation=operation,
        error_type=type(error).__name__,
        message=str(error),
    )


def _build_workspace(snapshot: PullRequestSnapshot) -> ReviewWorkspace:
    entries = [
        ManifestFile(
            path=file.path,
            status=file.status,
            additions=file.additions,
            deletions=file.deletions,
            changes=file.additions + file.deletions,
            diff_path=f"/diffs/{file.path}.diff",
            previous_path=file.previous_path,
        )
        for file in snapshot.files
    ]
    manifest = Manifest(
        repo=snapshot.repo,
        pull_request_number=snapshot.number,
        base_sha=snapshot.base_sha,
        head_sha=snapshot.head_sha,
        files=entries,
    )
    files = {
        entry.diff_path: snapshot_file.patch
        for entry, snapshot_file in zip(entries, snapshot.files, strict=True)
    }
    files["/manifest.json"] = manifest.model_dump_json(indent=2)
    workspace = ReviewWorkspace(manifest=manifest, files=files)
    logger.info(
        "manifest.workspace.ready repo=%s pr=%d virtual_files=%d",
        manifest.repo,
        manifest.pull_request_number,
        len(files),
    )
    logger.debug("manifest.workspace.paths %s", sorted(files))
    logger.debug("manifest.workspace.index %s", manifest.model_dump(mode="json"))
    return workspace


def create_manifest_review_agent(
    *,
    source: SnapshotSource,
    model: str | BaseChatModel,
    instructions: str = "",
) -> ManifestReviewAgent:
    """Create the production Manifest Review workflow used by Chat."""
    return ManifestReviewAgent(
        source=source,
        runner=ManifestReviewRunner(model=model, instructions=instructions),
    )
