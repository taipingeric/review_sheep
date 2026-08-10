"""The structured Review path over a fixed local Git checkout."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast

from deepagents import create_deep_agent as _create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict

from review_sheep.checkout import git_changed_files, git_diff
from review_sheep.domain import (
    Finding,
    Lens,
    Review,
    ReviewCheckout,
    ReviewError,
    ReviewOperation,
)

_PARALLEL_TOOL_USE_PROMPT = (
    "When multiple independent tool calls are needed, call them in parallel in the "
    "same assistant message. Keep dependent tool calls sequential. "
)

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


class CorrectnessFinding(Finding):
    """Finding contract fixed to the correctness Lens."""

    lens: Literal[Lens.CORRECTNESS] = Lens.CORRECTNESS


class SecurityFinding(Finding):
    """Finding contract fixed to the security Lens."""

    lens: Literal[Lens.SECURITY] = Lens.SECURITY


class ConventionsAndTestsFinding(Finding):
    """Finding contract fixed to the conventions-and-tests Lens."""

    lens: Literal[Lens.CONVENTIONS_AND_TESTS] = Lens.CONVENTIONS_AND_TESTS


class CorrectnessResult(BaseModel):
    """Structured output contract for the correctness Lens."""

    model_config = ConfigDict(frozen=True)

    findings: list[CorrectnessFinding]


class SecurityResult(BaseModel):
    """Structured output contract for the security Lens."""

    model_config = ConfigDict(frozen=True)

    findings: list[SecurityFinding]


class ConventionsAndTestsResult(BaseModel):
    """Structured output contract for the conventions-and-tests Lens."""

    model_config = ConfigDict(frozen=True)

    findings: list[ConventionsAndTestsFinding]


_LENS_RESULT_SCHEMAS: dict[Lens, type[BaseModel]] = {
    Lens.CORRECTNESS: CorrectnessResult,
    Lens.SECURITY: SecurityResult,
    Lens.CONVENTIONS_AND_TESTS: ConventionsAndTestsResult,
}

_LENS_SYSTEM_PROMPTS = {
    Lens.CORRECTNESS: (
        "Plan the Review, then inspect the whole pull request through the "
        "correctness Lens, not one file in isolation. Trace changed contracts "
        "across callers and callees, and report only actionable defects. Every "
        "output item must use the correctness lens."
    ),
    Lens.SECURITY: (
        "Plan the Review, then inspect the whole pull request through the security "
        "Lens. Trace data, identity, authorization, and trust boundaries across "
        "files. Report only actionable security defects. Every output item must "
        "use the security lens."
    ),
    Lens.CONVENTIONS_AND_TESTS: (
        "Plan the Review, then inspect the whole pull request through the "
        "conventions-and-tests Lens. Compare related implementation and tests "
        "across files. Report only actionable convention or test defects. Every "
        "output item must use the conventions-and-tests lens."
    ),
}


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
        findings: list[Finding] = []
        for lens in Lens:
            findings.extend(self._run_lens(lens, checkout))
        return findings

    def _run_lens(self, lens: Lens, checkout: ReviewCheckout) -> list[Finding]:
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
            response_format=ToolStrategy(_LENS_RESULT_SCHEMAS[lens]),
            system_prompt=(
                _PARALLEL_TOOL_USE_PROMPT
                + _CHECKOUT_PROMPT
                + _LENS_SYSTEM_PROMPTS[lens]
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
        return _validate_lens_findings(lens, result["structured_response"])


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


def _validate_lens_findings(lens: Lens, payload: Any) -> list[Finding]:
    findings: list[Finding] = []
    if lens is Lens.CORRECTNESS:
        findings.extend(CorrectnessResult.model_validate(payload).findings)
    elif lens is Lens.SECURITY:
        findings.extend(SecurityResult.model_validate(payload).findings)
    else:
        findings.extend(ConventionsAndTestsResult.model_validate(payload).findings)
    return findings


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
            checkout = source.prepare_checkout(
                repo=state["repo"],
                number=state["pull_request_number"],
            )
        except Exception as error:  # noqa: BLE001 - failures are public data here
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
            findings = runner.run(state["checkout"])
        except Exception as error:  # noqa: BLE001 - failures are public data here
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
