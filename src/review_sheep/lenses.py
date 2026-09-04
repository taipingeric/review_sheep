"""Shared Lens prompts and structured Finding contracts."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import Context, copy_context
from functools import cache
from typing import Any, Literal, overload

from pydantic import BaseModel, ConfigDict, create_model

from review_sheep.domain import (
    CustomLens,
    Finding,
    Lens,
    validate_custom_lenses,
)

logger = logging.getLogger(__name__)

PARALLEL_TOOL_USE_PROMPT = (
    "When multiple independent tool calls are needed, call them in parallel in the "
    "same assistant message. Keep dependent tool calls sequential. "
)

ReviewLens = Lens | CustomLens


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


LENS_RESULT_SCHEMAS: dict[Lens, type[BaseModel]] = {
    Lens.CORRECTNESS: CorrectnessResult,
    Lens.SECURITY: SecurityResult,
    Lens.CONVENTIONS_AND_TESTS: ConventionsAndTestsResult,
}

LENS_SYSTEM_PROMPTS = {
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


def lens_identifier(lens: ReviewLens) -> str:
    """Return the stable identity used for prompts, logs, and Findings."""
    return lens.value if isinstance(lens, Lens) else lens.identifier


def lens_system_prompt(lens: ReviewLens) -> str:
    """Return the concern-specific system prompt for one Lens."""
    if isinstance(lens, Lens):
        return LENS_SYSTEM_PROMPTS[lens]
    return (
        "Plan the Review, then inspect the whole pull request through the "
        f"{lens.identifier} Lens. {lens.instructions} Every output item must use "
        f"the {lens.identifier} lens."
    )


def lens_result_schema(lens: ReviewLens) -> type[Any]:
    """Build the structured output schema for one built-in or custom Lens."""
    if isinstance(lens, Lens):
        return LENS_RESULT_SCHEMAS[lens]
    return _custom_lens_result_schema(lens.identifier)


@cache
def _custom_lens_result_schema(identifier: str) -> type[Any]:
    finding_model: Any = create_model(
        f"{_schema_name(identifier)}Finding",
        __base__=Finding,
        lens=(Literal[identifier], identifier),
    )
    return create_model(
        f"{_schema_name(identifier)}Result",
        __config__=ConfigDict(frozen=True),
        findings=(list[finding_model], ...),
    )


def _schema_name(identifier: str) -> str:
    return "".join(part.capitalize() for part in identifier.split("-"))


@overload
def run_lenses_in_parallel(
    run_lens: Callable[[Lens], list[Finding]],
) -> list[Finding]: ...


@overload
def run_lenses_in_parallel(
    run_lens: Callable[[ReviewLens], list[Finding]],
    custom_lenses: Sequence[CustomLens],
) -> list[Finding]: ...


def run_lenses_in_parallel(
    run_lens: Callable[[Any], list[Finding]],
    custom_lenses: Sequence[CustomLens] = (),
) -> list[Finding]:
    """Run independent Lenses concurrently and flatten in stable order."""
    lenses: list[ReviewLens] = [*Lens, *validate_custom_lenses(custom_lenses)]
    logger.info("lenses.parallel.start count=%d", len(lenses))

    def run_with_context(context: Context, lens: ReviewLens) -> list[Finding]:
        return context.run(run_lens, lens)

    with ThreadPoolExecutor(
        max_workers=len(lenses),
        thread_name_prefix="review-sheep-lens",
    ) as executor:
        futures: list[Future[list[Finding]]] = [
            executor.submit(run_with_context, copy_context(), lens) for lens in lenses
        ]
        findings = [finding for future in futures for finding in future.result()]
    logger.info("lenses.parallel.complete findings=%d", len(findings))
    return findings


def validate_lens_findings(lens: ReviewLens, payload: Any) -> list[Finding]:
    """Validate model output against the schema fixed to one Lens."""
    return lens_result_schema(lens).model_validate(payload).findings
