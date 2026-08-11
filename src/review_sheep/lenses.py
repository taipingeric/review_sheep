"""Shared Lens prompts and structured Finding contracts."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from review_sheep.domain import Finding, Lens

logger = logging.getLogger(__name__)

PARALLEL_TOOL_USE_PROMPT = (
    "When multiple independent tool calls are needed, call them in parallel in the "
    "same assistant message. Keep dependent tool calls sequential. "
)


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


def run_lenses_in_parallel(
    run_lens: Callable[[Lens], list[Finding]],
) -> list[Finding]:
    """Run independent Lenses concurrently and flatten in stable Lens order."""
    lenses = list(Lens)
    logger.info("lenses.parallel.start count=%d", len(lenses))
    with ThreadPoolExecutor(
        max_workers=len(lenses),
        thread_name_prefix="review-sheep-lens",
    ) as executor:
        futures: dict[Lens, Future[list[Finding]]] = {
            lens: executor.submit(copy_context().run, run_lens, lens) for lens in lenses
        }
        findings = [finding for lens in lenses for finding in futures[lens].result()]
    logger.info("lenses.parallel.complete findings=%d", len(findings))
    return findings


def validate_lens_findings(lens: Lens, payload: Any) -> list[Finding]:
    """Validate model output against the schema fixed to one Lens."""
    findings: list[Finding] = []
    if lens is Lens.CORRECTNESS:
        findings.extend(CorrectnessResult.model_validate(payload).findings)
    elif lens is Lens.SECURITY:
        findings.extend(SecurityResult.model_validate(payload).findings)
    else:
        findings.extend(ConventionsAndTestsResult.model_validate(payload).findings)
    return findings
