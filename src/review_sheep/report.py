"""Pure human-facing rendering for structured Reviews."""

from __future__ import annotations

import logging

from review_sheep.domain import Finding, Lens, Location, Report, Review

logger = logging.getLogger(__name__)


def render_report(review: Review) -> Report:
    """Render one Review without changing its structured Findings."""
    logger.info(
        "report.render repo=%s pr=%d findings=%d",
        review.repo,
        review.pull_request_number,
        len(review.findings),
    )
    lines = [
        f"# Review Report: {review.repo}#{review.pull_request_number}",
        "",
        f"Head: `{review.head_sha}`",
        f"Findings: {len(review.findings)}",
        "",
    ]
    if not review.findings:
        lines.append("No Findings.")
        return Report(text="\n".join(lines) + "\n")

    for index, finding in enumerate(review.findings, start=1):
        lines.extend(_render_finding(index, finding))
    return Report(text="\n".join(lines))


def _render_finding(index: int, finding: Finding) -> list[str]:
    return [
        f"## {index}. {finding.severity.value.upper()} — {finding.description}",
        "",
        f"- Location: {_render_location(finding.location)}",
        f"- Confidence: {finding.confidence.value}",
        f"- Lens: {_render_lens(finding.lens)}",
        "",
    ]


def _render_lens(lens: Lens | str) -> str:
    return lens.value if isinstance(lens, Lens) else lens


def _render_location(location: Location) -> str:
    if location.path is None:
        return "pull request"
    if location.start_line is None:
        return f"`{location.path}` (whole file)"
    if location.start_line == location.end_line:
        return f"`{location.path}:{location.start_line}`"
    return f"`{location.path}:{location.start_line}-{location.end_line}`"
