"""The structured Review path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

from deepagents import create_deep_agent as _create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

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

_SERIAL_TOOL_USE_PROMPT = (
    "Call at most one tool in each assistant message; never call tools in parallel. "
)


class SnapshotSource(Protocol):
    """Fetch one stable pull-request snapshot."""

    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot: ...


class ReviewRunner(Protocol):
    """Review one prepared workspace through every configured Lens."""

    def run(self, workspace: ReviewWorkspace) -> list[Finding]: ...


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


class DeepAgentReviewRunner:
    """Run one independent deep subagent per Lens over the same workspace."""

    def __init__(
        self, *, subagents: Mapping[Lens, Any], instructions: str = ""
    ) -> None:
        configured = set(subagents)
        required = set(Lens)
        if configured != required:
            missing = ", ".join(sorted(lens.value for lens in required - configured))
            extra = ", ".join(sorted(lens.value for lens in configured - required))
            raise ValueError(
                f"DeepAgentReviewRunner requires every Lens; missing={missing!r}, "
                f"extra={extra!r}"
            )
        self._subagents = dict(subagents)
        self._instructions = instructions.strip()

    def run(self, workspace: ReviewWorkspace) -> list[Finding]:
        """Validate and concatenate raw Lens Findings in stable Lens order."""
        findings: list[Finding] = []
        for lens in Lens:
            findings.extend(self._run_lens(lens, workspace))
        return findings

    def _run_lens(self, lens: Lens, workspace: ReviewWorkspace) -> list[Finding]:
        request = (
            "Review the pull request snapshot in /manifest.json and its diff files "
            f"through the {lens.value} Lens."
        )
        if self._instructions:
            request += f"\n\nCaller instructions:\n{self._instructions}"
        state = {
            "messages": [
                {
                    "role": "user",
                    "content": request,
                }
            ],
            "files": {
                path: create_file_data(content)
                for path, content in workspace.files.items()
            },
        }
        result = self._subagents[lens].invoke(state)
        return _validate_lens_findings(lens, result["structured_response"])


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
    """Turn one pull-request snapshot into a structured Review."""

    def __init__(self, *, source: SnapshotSource, runner: ReviewRunner) -> None:
        self._source = source
        self._runner = runner

    def review(self, *, repo: str, number: int) -> Review | ReviewError:
        """Run one structured Review without writing to GitHub."""
        try:
            snapshot = self._source.fetch_snapshot(repo=repo, number=number)
        except Exception as error:  # noqa: BLE001 - failures are public data here
            return _review_error(
                repo=repo,
                number=number,
                operation=ReviewOperation.FETCH_SNAPSHOT,
                error=error,
            )

        try:
            workspace = _build_workspace(snapshot)
        except Exception as error:  # noqa: BLE001 - failures are public data here
            return _review_error(
                repo=repo,
                number=number,
                operation=ReviewOperation.PREPARE_WORKSPACE,
                error=error,
            )

        try:
            findings = self._runner.run(workspace)
        except Exception as error:  # noqa: BLE001 - failures are public data here
            return _review_error(
                repo=repo,
                number=number,
                operation=ReviewOperation.RUN_REVIEW,
                error=error,
            )
        return Review(
            repo=snapshot.repo,
            pull_request_number=snapshot.number,
            head_sha=snapshot.head_sha,
            manifest=workspace.manifest,
            findings=findings,
        )


def _review_error(
    *, repo: str, number: int, operation: ReviewOperation, error: Exception
) -> ReviewError:
    return ReviewError(
        repo=repo,
        pull_request_number=number,
        operation=operation,
        error_type=type(error).__name__,
        message=str(error),
    )


def _build_workspace(snapshot: PullRequestSnapshot) -> ReviewWorkspace:
    manifest_files = [
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
        head_sha=snapshot.head_sha,
        files=manifest_files,
    )
    files = {
        entry.diff_path: snapshot_file.patch
        for entry, snapshot_file in zip(
            manifest_files, snapshot.files, strict=True
        )
    }
    files["/manifest.json"] = manifest.model_dump_json(indent=2)
    return ReviewWorkspace(manifest=manifest, files=files)


def create_review_agent(
    *, source: SnapshotSource, runner: ReviewRunner
) -> ReviewAgent:
    """Create the Review module from explicit adapters."""
    return ReviewAgent(source=source, runner=runner)


def create_deep_review_agent(
    *,
    source: SnapshotSource,
    model: str | BaseChatModel,
    instructions: str = "",
) -> ReviewAgent:
    """Create the production Review workflow with one subagent per Lens."""
    correctness_subagent = _create_deep_agent(
        model=model,
        backend=StateBackend(),
        response_format=ToolStrategy(CorrectnessResult),
        system_prompt=(
            _SERIAL_TOOL_USE_PROMPT
            + "Plan the Review, then inspect the whole pull request through the "
            "correctness Lens, not one file in isolation. Start with /manifest.json, "
            "inspect every relevant diff under /diffs, trace changed contracts across "
            "callers and callees, and report only actionable defects. Every output "
            "item must use the correctness lens."
        ),
    )
    security_subagent = _create_deep_agent(
        model=model,
        backend=StateBackend(),
        response_format=ToolStrategy(SecurityResult),
        system_prompt=(
            _SERIAL_TOOL_USE_PROMPT
            + "Plan the Review, then inspect the whole pull request through the security "
            "Lens. Start with /manifest.json, inspect every relevant diff under /diffs, "
            "and trace data, identity, authorization, and trust boundaries across files. "
            "Report only actionable security defects. Every output item must use the "
            "security lens."
        ),
    )
    conventions_and_tests_subagent = _create_deep_agent(
        model=model,
        backend=StateBackend(),
        response_format=ToolStrategy(ConventionsAndTestsResult),
        system_prompt=(
            _SERIAL_TOOL_USE_PROMPT
            + "Plan the Review, then inspect the whole pull request through the "
            "conventions-and-tests Lens. Start with /manifest.json, inspect every "
            "relevant diff under /diffs, and compare related implementation and tests "
            "across files. Report only actionable convention or test defects. Every "
            "output item must use the conventions-and-tests lens."
        ),
    )
    return create_review_agent(
        source=source,
        runner=DeepAgentReviewRunner(
            subagents={
                Lens.CORRECTNESS: correctness_subagent,
                Lens.SECURITY: security_subagent,
                Lens.CONVENTIONS_AND_TESTS: conventions_and_tests_subagent,
            },
            instructions=instructions,
        ),
    )
