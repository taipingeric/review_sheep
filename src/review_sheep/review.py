"""The structured Review path over a fixed local Git checkout."""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any, NotRequired, Protocol, TypedDict, cast

from deepagents import create_deep_agent as _create_deep_agent
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, set_config_context
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from review_sheep.backends import ReviewFilesystemBackend
from review_sheep.brief import BRIEF_SYSTEM_PROMPT, verify_brief
from review_sheep.checkout import git_changed_files, git_diff
from review_sheep.domain import (
    CustomLens,
    Finding,
    PullRequestContext,
    Review,
    ReviewBrief,
    ReviewCheckout,
    ReviewContext,
    ReviewError,
    ReviewOperation,
    validate_custom_lenses,
    validate_review_contexts,
)
from review_sheep.lenses import (
    PARALLEL_TOOL_USE_PROMPT,
    LensExecutionError,
    ReviewLens,
    lens_identifier,
    lens_result_schema,
    lens_system_prompt,
    pull_request_description_prompt,
    review_context_prompt,
    run_lenses_in_parallel,
    validate_lens_findings,
)
from review_sheep.merge import merge_findings
from review_sheep.tracing import lens_run_config

logger = logging.getLogger(__name__)

_CHECKOUT_PROMPT = (
    "The repository is a clean local worktree fixed at the pull request head SHA. "
    "Start by calling list_changed_files, then call get_diff for relevant changed "
    "paths. Prefer exact paths from those tools. Use read_file, grep, and scoped "
    "glob patterns to inspect complete files and trace unchanged callers, callees, "
    "tests, and configuration. Never use a repository-root broad glob such as '*', "
    "'**', or '**/*'; narrow the directory or pattern before searching. Treat the "
    "worktree as read-only; never call write_file, edit_file, or delete_file. "
)


class ReviewCheckoutSource(Protocol):
    """Prepare one verified local checkout for a pull request."""

    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout: ...


class DisposableReviewCheckoutSource(ReviewCheckoutSource, Protocol):
    """A checkout source that owns a disposable checkout lifecycle."""

    def cleanup_checkout(self, checkout: ReviewCheckout) -> None: ...


class ReviewRunner(Protocol):
    """Review one fixed checkout through every configured Lens."""

    def run(self, checkout: ReviewCheckout) -> list[Finding]: ...


class _ReviewGraphState(TypedDict):
    """Internal state for the deterministic Review LangGraph."""

    repo: str
    pull_request_number: int
    checkout: NotRequired[ReviewCheckout]
    findings: NotRequired[list[Finding]]
    lens_error: NotRequired[ReviewError]
    brief: NotRequired[ReviewBrief]
    brief_error: NotRequired[str]
    result: NotRequired[Review | ReviewError]


class DeepAgentReviewRunner:
    """Run one independent deep agent per Lens over the same fixed checkout."""

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        instructions: str = "",
        tools: Sequence[BaseTool] = (),
        custom_lenses: Sequence[CustomLens] = (),
        lens_concurrency: int | None = None,
        context_providers: Sequence[ReviewContext] = (),
        brief: bool = True,
    ) -> None:
        self._brief = brief
        self._model = model
        self._instructions = instructions.strip()
        self._tools = tuple(tools)
        self._custom_lenses = validate_custom_lenses(custom_lenses)
        self._lens_concurrency = lens_concurrency
        self._contexts = validate_review_contexts(context_providers)
        # The Lenses and the brief run in parallel on one checkout; fetch the
        # caller context once for both. Entries vanish with the checkout.
        self._context_cache: weakref.WeakKeyDictionary[ReviewCheckout, str] = (
            weakref.WeakKeyDictionary()
        )
        self._context_lock = threading.Lock()

    def _caller_context(self, checkout: ReviewCheckout) -> str:
        with self._context_lock:
            if checkout not in self._context_cache:
                self._context_cache[checkout] = review_context_prompt(
                    self._contexts, _pull_request_context(checkout)
                )
            return self._context_cache[checkout]

    def run(
        self,
        checkout: ReviewCheckout,
        *,
        config: RunnableConfig | None = None,
    ) -> list[Finding]:
        """Run every Lens, then merge Findings that share one root cause."""
        logger.info(
            "checkout.review.start repo=%s pr=%d base=%s head=%s",
            checkout.repo,
            checkout.pull_request_number,
            checkout.base_sha,
            checkout.head_sha,
        )
        effective_config = ensure_config(config)
        caller_context = self._caller_context(checkout)
        findings = run_lenses_in_parallel(
            lambda lens: self._run_lens(
                lens, checkout, caller_context, config=effective_config
            ),
            self._custom_lenses,
            lens_concurrency=self._lens_concurrency,
        )
        findings = merge_findings(findings, model=self._model, config=effective_config)
        logger.info("checkout.review.complete findings=%d", len(findings))
        return findings

    @property
    def brief_enabled(self) -> bool:
        """Whether Reviews also carry a Review Brief."""
        return self._brief

    def brief(
        self,
        checkout: ReviewCheckout,
        *,
        config: RunnableConfig | None = None,
    ) -> ReviewBrief:
        """Write Intent, Acceptance Criteria, and Decisions with one agent.

        The brief agent never sees the Findings, so its verdicts are independent
        of the Lenses that wrote them.
        """
        logger.info("checkout.brief.start")
        effective_config = ensure_config(config)
        agent = self._create_agent(
            checkout,
            response_format=ReviewBrief,
            system_prompt=PARALLEL_TOOL_USE_PROMPT
            + _CHECKOUT_PROMPT
            + BRIEF_SYSTEM_PROMPT,
        )
        request = (
            f"Write the Review Brief for {checkout.repo}#{checkout.pull_request_number} "
            f"from {checkout.base_sha}...{checkout.head_sha}."
        )
        request += pull_request_description_prompt(
            title=checkout.title, body=checkout.body
        )
        request += self._caller_context(checkout)
        if self._instructions:
            request += f"\n\nCaller instructions:\n{self._instructions}"
        state = cast(Any, {"messages": [{"role": "user", "content": request}]})
        with set_config_context(
            lens_run_config(config=effective_config, lens="brief")
        ) as context:
            result = context.run(agent.invoke, state)
        brief = ReviewBrief.model_validate(result["structured_response"])
        logger.info(
            "checkout.brief.complete criteria=%d decisions=%d",
            len(brief.acceptance_criteria),
            len(brief.decisions),
        )
        return brief

    def _create_agent(
        self,
        checkout: ReviewCheckout,
        *,
        response_format: type[Any],
        system_prompt: str,
    ) -> Any:
        return _create_deep_agent(
            model=self._model,
            tools=[*_checkout_tools(checkout), *self._tools],
            backend=ReviewFilesystemBackend(
                root_dir=checkout.root,
                virtual_mode=True,
            ),
            permissions=[
                FilesystemPermission(
                    operations=["write"],
                    paths=["/**"],
                    mode="deny",
                )
            ],
            response_format=ToolStrategy(response_format),
            system_prompt=system_prompt,
        )

    def _run_lens(
        self,
        lens: ReviewLens,
        checkout: ReviewCheckout,
        caller_context: str,
        *,
        config: RunnableConfig | None = None,
    ) -> list[Finding]:
        identifier = lens_identifier(lens)
        logger.info("checkout.lens.start lens=%s", identifier)
        agent = self._create_agent(
            checkout,
            response_format=lens_result_schema(lens),
            system_prompt=(
                PARALLEL_TOOL_USE_PROMPT + _CHECKOUT_PROMPT + lens_system_prompt(lens)
            ),
        )
        request = (
            f"Review {checkout.repo}#{checkout.pull_request_number} from "
            f"{checkout.base_sha}...{checkout.head_sha} through the "
            f"{identifier} Lens."
        )
        request += pull_request_description_prompt(
            title=checkout.title, body=checkout.body
        )
        request += caller_context
        if self._instructions:
            request += f"\n\nCaller instructions:\n{self._instructions}"
        state = cast(Any, {"messages": [{"role": "user", "content": request}]})
        with set_config_context(
            lens_run_config(config=config, lens=identifier)
        ) as context:
            result = context.run(agent.invoke, state)
        findings = validate_lens_findings(lens, result["structured_response"])
        logger.info(
            "checkout.lens.complete lens=%s findings=%d",
            identifier,
            len(findings),
        )
        logger.debug(
            "checkout.lens.findings lens=%s data=%s",
            identifier,
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


def _pull_request_context(checkout: ReviewCheckout) -> PullRequestContext:
    return PullRequestContext(
        repo=checkout.repo,
        number=checkout.pull_request_number,
        base_sha=checkout.base_sha,
        head_sha=checkout.head_sha,
        title=checkout.title,
        body=checkout.body,
        head_ref=checkout.head_ref,
    )


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

    def review(
        self,
        *,
        repo: str,
        number: int,
        config: RunnableConfig | None = None,
    ) -> Review | ReviewError:
        """Run one structured Review without writing to GitHub or the checkout."""
        state = cast(
            _ReviewGraphState,
            self._graph.invoke(
                _ReviewGraphState(repo=repo, pull_request_number=number),
                config=ensure_config(config),
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

    def run_lenses(state: _ReviewGraphState) -> dict[str, object]:
        try:
            logger.info("checkout.graph.run_review")
            findings = runner.run(state["checkout"])
        except Exception as error:
            logger.exception("checkout.graph.run_review.failed")
            return {
                "lens_error": _review_error(
                    state=state,
                    operation=ReviewOperation.RUN_REVIEW,
                    error=error,
                )
            }
        return {"findings": findings}

    def write_brief(state: _ReviewGraphState) -> dict[str, object]:
        brief, brief_error = _run_brief(runner, state["checkout"])
        return {"brief": brief, "brief_error": brief_error}

    def assemble(state: _ReviewGraphState) -> dict[str, object]:
        checkout = state["checkout"]
        try:
            if "lens_error" in state:
                return {"result": state["lens_error"]}
            brief = state["brief"]
            return {
                "result": Review(
                    repo=checkout.repo,
                    pull_request_number=checkout.pull_request_number,
                    base_sha=checkout.base_sha,
                    head_sha=checkout.head_sha,
                    findings=state["findings"],
                    **brief.model_dump(),
                    brief_error=state["brief_error"],
                )
            }
        finally:
            _cleanup_checkout(source, checkout)

    def route(state: _ReviewGraphState) -> list[str]:
        # Lenses and the brief are independent, so they run in parallel.
        if "result" in state:
            return [END]
        return ["run_lenses", "write_brief"]

    graph = StateGraph(_ReviewGraphState)
    graph.add_node("prepare_checkout", prepare_checkout)
    graph.add_node("run_lenses", run_lenses)
    graph.add_node("write_brief", write_brief)
    graph.add_node("assemble", assemble)
    graph.add_edge(START, "prepare_checkout")
    graph.add_conditional_edges("prepare_checkout", route)
    graph.add_edge(["run_lenses", "write_brief"], "assemble")
    graph.add_edge("assemble", END)
    return graph.compile()


def _run_brief(
    runner: ReviewRunner, checkout: ReviewCheckout
) -> tuple[ReviewBrief, str]:
    """Ask an opted-in runner for a Review Brief.

    A failure leaves the brief empty and returns the reason, so a reader can
    tell "failed" from "not asked for".
    """
    if not getattr(runner, "brief_enabled", False):
        return ReviewBrief(), ""
    try:
        return verify_brief(cast(Any, runner).brief(checkout), checkout.root), ""
    except Exception as error:
        logger.exception("checkout.graph.brief.failed")
        return ReviewBrief(), f"{type(error).__name__}: {error}"


def _cleanup_checkout(source: ReviewCheckoutSource, checkout: ReviewCheckout) -> None:
    cleanup = getattr(source, "cleanup_checkout", None)
    if cleanup is None:
        return
    try:
        cleanup(checkout)
    except Exception:
        logger.exception("checkout.graph.cleanup.failed")


def _review_error(
    *, state: _ReviewGraphState, operation: ReviewOperation, error: Exception
) -> ReviewError:
    lens = error.lens if isinstance(error, LensExecutionError) else None
    cause = error.cause if isinstance(error, LensExecutionError) else error
    return ReviewError(
        repo=state["repo"],
        pull_request_number=state["pull_request_number"],
        operation=operation,
        error_type=type(cause).__name__,
        message=str(error),
        lens=lens,
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
    tools: Sequence[BaseTool] = (),
    custom_lenses: Sequence[CustomLens] = (),
    lens_concurrency: int | None = None,
    context_providers: Sequence[ReviewContext] = (),
    brief: bool = True,
) -> ReviewAgent:
    """Create the production Review workflow with one deep agent per Lens.

    By default one extra, independent agent also writes Intent, Acceptance
    Criteria, and Decisions onto the Review. Pass ``brief=False`` to skip it.
    """
    return create_review_agent(
        source=source,
        runner=DeepAgentReviewRunner(
            model=model,
            instructions=instructions,
            tools=tools,
            custom_lenses=custom_lenses,
            lens_concurrency=lens_concurrency,
            context_providers=context_providers,
            brief=brief,
        ),
    )
