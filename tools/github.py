"""GitHub pull request tools."""

import json
import os
from functools import lru_cache

from github import Auth, Github
from langchain.tools import tool

from tools._errors import errors_as_data

MAX_LIMIT = 50
MAX_FILES = 50
# Whole diff for one file, fetched by get_file_patch.
PATCH_CHUNK_CHARS = 12_000
# Preview only, so a multi-file listing stays small.
MAX_PATCH_CHARS = 3000


@lru_cache(maxsize=1)
def _client() -> Github:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is missing from .env")
    return Github(auth=Auth.Token(token))


def _resolve_repo(repo: str) -> str:
    target = repo or os.getenv("GITHUB_REPO")
    if not target:
        raise RuntimeError("no repository given and GITHUB_REPO is missing from .env")
    return target


_github_errors_as_data = errors_as_data(repo_resolver=_resolve_repo)


def _summarize(pull) -> dict:
    return {
        "number": pull.number,
        "title": pull.title,
        "state": pull.state,
        "author": pull.user.login if pull.user else None,
        "created_at": pull.created_at.isoformat() if pull.created_at else None,
        "updated_at": pull.updated_at.isoformat() if pull.updated_at else None,
        "url": pull.html_url,
    }


@tool
@_github_errors_as_data
def list_pull_requests(state: str = "open", limit: int = 10, repo: str = "") -> str:
    """List pull requests for a GitHub repository, most recently updated first.

    Args:
        state: Which pull requests to list: "open", "closed", or "all".
        limit: Maximum number of pull requests to return (1-50).
        repo: Repository in "payfazz/name" form. Omit to use the default repository.
    """
    if state not in {"open", "closed", "all"}:
        return json.dumps({"error": f"invalid state {state!r}; use open, closed, or all"})

    limit = max(1, min(limit, MAX_LIMIT))
    target = _resolve_repo(repo)

    repository = _client().get_repo(target)
    pulls = repository.get_pulls(state=state, sort="updated", direction="desc")
    rows = [_summarize(pull) for pull in pulls[:limit]]

    return json.dumps(
        {"repo": target, "state": state, "count": len(rows), "pull_requests": rows}
    )


@tool
@_github_errors_as_data
def get_pull_request(number: int, repo: str = "") -> str:
    """Get details for one pull request, including its description and review state.

    Args:
        number: The pull request number.
        repo: Repository in "owner/name" form. Omit to use the default repository.
    """
    target = _resolve_repo(repo)
    pull = _client().get_repo(target).get_pull(number)

    details = _summarize(pull)
    details.update(
        {
            "repo": target,
            "body": (pull.body or "")[:2000],
            "base": pull.base.ref,
            "head": pull.head.ref,
            "draft": pull.draft,
            "merged": pull.merged,
            "mergeable_state": pull.mergeable_state,
            "changed_files": pull.changed_files,
            "additions": pull.additions,
            "deletions": pull.deletions,
            "labels": [label.name for label in pull.labels],
        }
    )

    return json.dumps(details)


@tool
@_github_errors_as_data
def get_pull_request_files(number: int, repo: str = "", include_patch: bool = False) -> str:
    """List the files changed by a pull request, with per-file line counts.

    Args:
        number: The pull request number.
        repo: Repository in "owner/name" form. Omit to use the default repository.
        include_patch: Set True to include a short diff preview for each file.
            Previews are cut off; call get_file_patch to review a file.
    """
    target = _resolve_repo(repo)
    pull = _client().get_repo(target).get_pull(number)

    files = []
    for changed in pull.get_files()[:MAX_FILES]:
        entry = {
            "filename": changed.filename,
            "status": changed.status,
            "additions": changed.additions,
            "deletions": changed.deletions,
            "changes": changed.changes,
        }
        if changed.previous_filename:
            entry["previous_filename"] = changed.previous_filename
        if include_patch and changed.patch:
            entry["patch"] = changed.patch[:MAX_PATCH_CHARS]
            entry["patch_truncated"] = len(changed.patch) > MAX_PATCH_CHARS
        files.append(entry)

    return json.dumps(
        {
            "repo": target,
            "number": number,
            "changed_files": pull.changed_files,
            "additions": pull.additions,
            "deletions": pull.deletions,
            "returned": len(files),
            "truncated": pull.changed_files > len(files),
            "files": files,
        }
    )


@tool
@_github_errors_as_data
def get_pull_request_reviews(number: int, repo: str = "") -> str:
    """Get review status for a pull request: who approved, who requested changes, who is pending.

    Args:
        number: The pull request number.
        repo: Repository in "owner/name" form. Omit to use the default repository.
    """
    target = _resolve_repo(repo)
    pull = _client().get_repo(target).get_pull(number)

    reviews = []
    latest_by_user: dict[str, str] = {}
    for review in pull.get_reviews():
        author = review.user.login if review.user else None
        reviews.append(
            {
                "author": author,
                "state": review.state,
                "submitted_at": review.submitted_at.isoformat()
                if review.submitted_at
                else None,
                "body": (review.body or "")[:500],
            }
        )
        # A reviewer's effective verdict is their latest non-comment review.
        if author and review.state != "COMMENTED":
            latest_by_user[author] = review.state

    pending = [user.login for user in pull.get_review_requests()[0]]

    return json.dumps(
        {
            "repo": target,
            "number": number,
            "state": pull.state,
            "merged": pull.merged,
            "mergeable_state": pull.mergeable_state,
            "approved_by": [u for u, s in latest_by_user.items() if s == "APPROVED"],
            "changes_requested_by": [
                u for u, s in latest_by_user.items() if s == "CHANGES_REQUESTED"
            ],
            "review_requested_from": pending,
            "review_count": len(reviews),
            "reviews": reviews,
        }
    )


github_toolkit = [
    list_pull_requests,
    get_pull_request,
    get_pull_request_files,
    get_pull_request_reviews,
]
