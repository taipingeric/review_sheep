"""The structured Review path over a fixed local Git checkout."""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Literal, NotRequired, Protocol, TypedDict, cast

from deepagents import create_deep_agent as _create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from review_sheep.checkout import git_changed_files, git_diff
from review_sheep.domain import (
    Finding,
    Lens,
    Review,
    ReviewCheckout,
    ReviewError,
    ReviewOperation,
)
from review_sheep.lenses import (
    LENS_RESULT_SCHEMAS,
    LENS_SYSTEM_PROMPTS,
    PARALLEL_TOOL_USE_PROMPT,
    validate_lens_findings,
)

logger = logging.getLogger(__name__)

_CHECKOUT_PROMPT = (
    "The repository is a clean local worktree fixed at the pull request head SHA. "
    "Start by calling list_changed_files, then call get_diff for relevant changed "
    "paths. Use read_file, grep, and glob to inspect complete files and trace "
    "unchanged callers, callees, tests, and configuration. Treat the worktree as "
    "read-only; never call write_file, edit_file, or delete_file. "
)


class ReviewCheckoutSource(Protocol):
    """Prepare one verified local checkout for a pull request."""

    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout: ...


class ReviewRunner(Protocol):
    """Review one fixed checkout through every configured Lens."""

    def run(self, checkout: ReviewCheckout) -> list[Finding]: ...


class _ReviewGraphState(TypedDict):
    """Internal state for the deterministic Review LangGraph."""

    repo: str
    pull_request_number: int
    checkout: NotRequired[ReviewCheckout]
    result: NotRequired[Review | ReviewError]


class DeepAgentReviewRunner:
    """Run one independent deep agent per Lens over the same fixed checkout."""

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        instructions: str = "",
    ) -> None:
        self._model = model
        self._instructions = instructions.strip()

    def run(self, checkout: ReviewCheckout) -> list[Finding]:
        """Run and concatenate Lens Findings in stable Lens order."""
        logger.info(
            "checkout.review.start repo=%s pr=%d base=%s head=%s",
            checkout.repo,
            checkout.pull_request_number,
            checkout.base_sha,
            checkout.head_sha,
        )
        findings: list[Finding] = []
        for lens in Lens:
            findings.extend(self._run_lens(lens, checkout))
        logger.info("checkout.review.complete findings=%d", len(findings))
        return findings

    def _run_lens(self, lens: Lens, checkout: ReviewCheckout) -> list[Finding]:
        logger.info("checkout.lens.start lens=%s", lens.value)
        agent = _create_deep_agent(
            model=self._model,
            tools=_checkout_tools(checkout),
            backend=FilesystemBackend(root_dir=checkout.root, virtual_mode=True),
            permissions=[
                FilesystemPermission(
                    operations=["write"],
                    paths=["/**"],
                    mode="deny",
                )
            ],
            response_format=ToolStrategy(LENS_RESULT_SCHEMAS[lens]),
            system_prompt=(
                PARALLEL_TOOL_USE_PROMPT + _CHECKOUT_PROMPT + LENS_SYSTEM_PROMPTS[lens]
            ),
        )
        request = (
            f"Review {checkout.repo}#{checkout.pull_request_number} from "
            f"{checkout.base_sha}...{checkout.head_sha} through the "
            f"{lens.value} Lens."
        )
        if self._instructions:
            request += f"\n\nCaller instructions:\n{self._instructions}"
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        findings = validate_lens_findings(lens, result["structured_response"])
        logger.info(
            "checkout.lens.complete lens=%s findings=%d",
            lens.value,
            len(findings),
        )
        logger.debug(
            "checkout.lens.findings lens=%s data=%s",
            lens.value,
            [finding.model_dump(mode="json") for finding in findings],
        )
        return findings


def _checkout_tools(checkout: ReviewCheckout) -> list[BaseTool]:
    @tool
    def list_changed_files() -> str:
        """List PR-changed paths and Git status from base SHA to head SHA."""
        return git_changed_files(checkout) or "No changed files."

    @tool
    def get_diff(path: str = "") -> str:
        """Read the PR diff; optionally pass one repository-relative changed path."""
        normalized = _normalize_repo_path(path)
        return git_diff(checkout, path=normalized) or "No diff for that path."

    return [list_changed_files, get_diff]


def _normalize_repo_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/").lstrip("/")
    if not normalized:
        return ""
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise ValueError("diff path must stay inside the review checkout")
    return str(PurePosixPath(*parts))


class ReviewAgent:
    """Run the Review LangGraph behind one stable public interface."""

    def __init__(self, *, source: ReviewCheckoutSource, runner: ReviewRunner) -> None:
        self._graph = _create_review_graph(source=source, runner=runner)

    def review(self, *, repo: str, number: int) -> Review | ReviewError:
        """Run one structured Review without writing to GitHub or the checkout."""
        state = cast(
            _ReviewGraphState,
            self._graph.invoke(
                _ReviewGraphState(repo=repo, pull_request_number=number)
            ),
        )
        return state["result"]


def _create_review_graph(
    *, source: ReviewCheckoutSource, runner: ReviewRunner
) -> CompiledStateGraph[
    _ReviewGraphState,
    None,
    _ReviewGraphState,
    _ReviewGraphState,
]:
    """Compile checkout validation and Lens execution into one LangGraph."""

    def prepare_checkout(state: _ReviewGraphState) -> dict[str, object]:
        try:
            logger.info(
                "checkout.graph.prepare repo=%s pr=%d",
                state["repo"],
                state["pull_request_number"],
            )
            checkout = source.prepare_checkout(
                repo=state["repo"],
                number=state["pull_request_number"],
            )
        except Exception as error:
            logger.exception("checkout.graph.prepare.failed")
            return {
                "result": _review_error(
                    state=state,
                    operation=ReviewOperation.PREPARE_CHECKOUT,
                    error=error,
                )
            }
        return {"checkout": checkout}

    def run_review(state: _ReviewGraphState) -> dict[str, object]:
        try:
            logger.info("checkout.graph.run_review")
            findings = runner.run(state["checkout"])
        except Exception as error:
            logger.exception("checkout.graph.run_review.failed")
            return {
                "result": _review_error(
                    state=state,
                    operation=ReviewOperation.RUN_REVIEW,
                    error=error,
                )
            }
        checkout = state["checkout"]
        return {
            "result": Review(
                repo=checkout.repo,
                pull_request_number=checkout.pull_request_number,
                base_sha=checkout.base_sha,
                head_sha=checkout.head_sha,
                findings=findings,
            )
        }

    def route(state: _ReviewGraphState) -> Literal["continue", "stop"]:
        return "stop" if "result" in state else "continue"

    graph = StateGraph(_ReviewGraphState)
    graph.add_node("prepare_checkout", prepare_checkout)
    graph.add_node("run_review", run_review)
    graph.add_edge(START, "prepare_checkout")
    graph.add_conditional_edges(
        "prepare_checkout",
        route,
        {"continue": "run_review", "stop": END},
    )
    graph.add_edge("run_review", END)
    return graph.compile()


def _review_error(
    *, state: _ReviewGraphState, operation: ReviewOperation, error: Exception
) -> ReviewError:
    return ReviewError(
        repo=state["repo"],
        pull_request_number=state["pull_request_number"],
        operation=operation,
        error_type=type(error).__name__,
        message=str(error),
    )


def create_review_agent(
    *, source: ReviewCheckoutSource, runner: ReviewRunner
) -> ReviewAgent:
    """Create the Review module from explicit adapters."""
    return ReviewAgent(source=source, runner=runner)


def create_deep_review_agent(
    *,
    source: ReviewCheckoutSource,
    model: str | BaseChatModel,
    instructions: str = "",
) -> ReviewAgent:
    """Create the production Review workflow with one deep agent per Lens."""
    return create_review_agent(
        source=source,
        runner=DeepAgentReviewRunner(model=model, instructions=instructions),
    )
