"""Review Context builders that leave every external client to the caller."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence

from review_sheep.domain import PullRequestContext, ReviewContext

logger = logging.getLogger(__name__)

NOTION_PAGE_PATTERN = (
    r"https?://(?:[\w-]+\.)?notion\.(?:so|site)/[^\s?#)>\]]*?"
    r"(?<![0-9a-fA-F])([0-9a-fA-F]{32})(?![0-9a-fA-F])"
)
JIRA_ISSUE_URL_PATTERN = (
    r"https?://[\w.-]+\.atlassian\.net/(?:browse/|[^\s]*?[?&]selectedIssue=)"
    r"([A-Z][A-Z0-9_]+-\d+)(?!\d)"
)
_JIRA_PROJECT_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def find_references(
    pattern: str | re.Pattern[str],
    pull_request: PullRequestContext,
    *,
    normalize: Callable[[str], str] | None = None,
) -> list[str]:
    """Return unique references from the PR title, body, and head branch.

    A pattern with capturing groups yields the first group that matched;
    otherwise it yields the whole match. ``normalize`` runs before duplicates
    are removed. References keep the order they first appear in.
    """
    compiled = re.compile(pattern)
    references: list[str] = []
    for text in (pull_request.title, pull_request.body, pull_request.head_ref):
        for match in compiled.finditer(text):
            reference = next(
                (group for group in match.groups() if group is not None),
                match.group(0),
            )
            if normalize is not None:
                reference = normalize(reference)
            if reference not in references:
                references.append(reference)
    return references


def linked_reference_context(
    *,
    identifier: str,
    pattern: str | re.Pattern[str],
    fetch: Callable[[str], str],
    max_references: int = 5,
    normalize: Callable[[str], str] | None = None,
) -> ReviewContext:
    """Build a Review Context that fetches each reference linked from the PR.

    The caller owns ``fetch``, including its client, credentials, and
    authorization. A failing or empty reference is skipped so the others still
    reach every Lens.
    """
    if (
        isinstance(max_references, bool)
        or not isinstance(max_references, int)
        or max_references < 1
    ):
        raise ValueError("max_references must be a positive integer")

    def collect(pull_request: PullRequestContext) -> str:
        sections: list[str] = []
        references = find_references(pattern, pull_request, normalize=normalize)[
            :max_references
        ]
        for reference in references:
            try:
                text = fetch(reference).strip()
            except Exception:
                logger.exception(
                    "review.context.reference_failed context=%s reference=%s",
                    identifier,
                    reference,
                )
                continue
            if text:
                sections.append(f"## {reference}\n{text}")
        return "\n\n".join(sections)

    return ReviewContext(identifier=identifier, fetch=collect)


def notion_page_context(
    fetch_page: Callable[[str], str],
    *,
    identifier: str = "notion",
    max_pages: int = 5,
) -> ReviewContext:
    """Build a Review Context from Notion page links in the PR.

    ``fetch_page`` receives a lower-case 32-character Notion page ID and returns
    the page text. Review Sheep never calls the Notion API itself.
    """
    return linked_reference_context(
        identifier=identifier,
        pattern=NOTION_PAGE_PATTERN,
        fetch=fetch_page,
        max_references=max_pages,
        normalize=str.lower,
    )


def jira_issue_context(
    fetch_issue: Callable[[str], str],
    *,
    project_keys: Sequence[str] = (),
    identifier: str = "jira",
    max_issues: int = 5,
) -> ReviewContext:
    """Build a Review Context from Jira issues linked in the PR.

    Jira Cloud links such as ``https://acme.atlassian.net/browse/ABC-123`` always
    match. A bare key such as ``ABC-123`` or ``abc-123-fix`` matches only for the
    listed ``project_keys``, which avoids false hits such as ``UTF-8``.
    ``fetch_issue`` receives the upper-case issue key and returns the issue
    text. Review Sheep never calls the Jira API itself.
    """
    pattern = JIRA_ISSUE_URL_PATTERN
    if project_keys:
        for key in project_keys:
            if not _JIRA_PROJECT_KEY_RE.fullmatch(key):
                raise ValueError(f"invalid Jira project key: {key!r}")
        keys = "|".join(re.escape(key) for key in project_keys)
        pattern += rf"|(?<![A-Za-z0-9])((?i:{keys})-\d+)(?!\d)"
    return linked_reference_context(
        identifier=identifier,
        pattern=pattern,
        fetch=fetch_issue,
        max_references=max_issues,
        normalize=str.upper,
    )
