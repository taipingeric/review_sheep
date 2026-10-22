"""Pure human-facing rendering for structured Reviews."""

from __future__ import annotations

import logging
from urllib.parse import quote

from review_sheep.domain import Evidence, Finding, Lens, Location, Report, Review

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
        lines.extend(_render_finding(index, finding, review))
    return Report(text="\n".join(lines))


def github_line_url(*, repo: str, head_sha: str, location: Location) -> str:
    """Link a head-commit Location to its lines on GitHub."""
    url = f"https://github.com/{repo}/blob/{head_sha}"
    if location.path is None:
        return url
    url += f"/{quote(location.path)}"
    if location.start_line is None:
        return url
    url += f"#L{location.start_line}"
    if location.end_line != location.start_line:
        url += f"-L{location.end_line}"
    return url


def _render_finding(index: int, finding: Finding, review: Review) -> list[str]:
    heading = finding.title or finding.description
    lines = [f"## {index}. {finding.severity.value.upper()} — {heading}", ""]
    if finding.title:
        lines.extend([finding.description, ""])
    lines.extend(
        [
            f"- Location: {_render_location(finding.location)}",
            f"- Confidence: {finding.confidence.value}",
            _render_lenses(finding),
        ]
    )
    if finding.evidence:
        lines.append("- Evidence:")
        lines.extend(_render_evidence(item, review) for item in finding.evidence)
    lines.append("")
    return lines


def _render_evidence(evidence: Evidence, review: Review) -> str:
    url = github_line_url(
        repo=review.repo, head_sha=review.head_sha, location=evidence.location
    )
    return f"  - [{_render_location(evidence.location)}]({url}) — {evidence.note}"


def _render_lenses(finding: Finding) -> str:
    if len(finding.lenses) == 1:
        return f"- Lens: {_render_lens(finding.lens)}"
    return "- Lenses: " + ", ".join(_render_lens(lens) for lens in finding.lenses)


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
