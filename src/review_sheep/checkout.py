"""Validate the fixed local Git checkout used by Review."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Protocol

from review_sheep.domain import PullRequestRevision, ReviewCheckout

logger = logging.getLogger(__name__)


class PullRequestRevisionSource(Protocol):
    """Read the base and head commits identifying a pull request."""

    def get_pull_request_revision(
        self, *, repo: str, number: int
    ) -> PullRequestRevision: ...


class GitCheckoutSource:
    """Prepare Review input from one clean, fixed-SHA local worktree."""

    def __init__(
        self, *, revisions: PullRequestRevisionSource, root: str | Path
    ) -> None:
        self._revisions = revisions
        self._root = Path(root).expanduser().resolve()

    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout:
        """Validate the worktree against the pull request's immutable revisions."""
        logger.info(
            "checkout.prepare.start repo=%s pr=%d root=%s",
            repo,
            number,
            self._root,
        )
        if not self._root.is_dir():
            raise RuntimeError(
                f"review checkout directory does not exist: {self._root}"
            )
        revision = self._revisions.get_pull_request_revision(repo=repo, number=number)
        root = Path(_git(self._root, "rev-parse", "--show-toplevel")).resolve()
        if root != self._root:
            raise RuntimeError(
                "review checkout path must be the Git worktree root: "
                f"expected {root}, received {self._root}"
            )
        head_sha = _git(root, "rev-parse", "HEAD")
        logger.info(
            "checkout.prepare.revisions expected_head=%s actual_head=%s base=%s",
            revision.head_sha,
            head_sha,
            revision.base_sha,
        )
        if head_sha != revision.head_sha:
            raise RuntimeError(
                "checkout HEAD does not match pull request head: "
                f"expected {revision.head_sha}, found {head_sha}"
            )
        _git(root, "cat-file", "-e", f"{revision.base_sha}^{{commit}}")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
        if dirty:
            raise RuntimeError("review checkout must be clean")
        checkout = ReviewCheckout(
            repo=revision.repo,
            pull_request_number=revision.number,
            base_sha=revision.base_sha,
            head_sha=revision.head_sha,
            root=root,
        )
        logger.info("checkout.prepare.complete clean=true")
        return checkout


def git_changed_files(checkout: ReviewCheckout) -> str:
    """Return a name-status index generated from the fixed checkout."""
    result = _git(
        checkout.root,
        "diff",
        "--name-status",
        "--find-renames",
        f"{checkout.base_sha}...{checkout.head_sha}",
    )
    logger.info(
        "checkout.changed_files repo=%s pr=%d entries=%d",
        checkout.repo,
        checkout.pull_request_number,
        len(result.splitlines()) if result else 0,
    )
    logger.debug("checkout.changed_files.index %s", result)
    return result


def git_diff(checkout: ReviewCheckout, *, path: str = "") -> str:
    """Return the pull-request diff, optionally restricted to one worktree path."""
    args = [
        "diff",
        "--find-renames",
        "--no-ext-diff",
        f"{checkout.base_sha}...{checkout.head_sha}",
    ]
    if path:
        args.extend(["--", path])
    result = _git(checkout.root, *args)
    logger.info(
        "checkout.diff path=%s characters=%d",
        path or "all",
        len(result),
    )
    return result


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("git executable is not available") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "git command failed"
        raise RuntimeError(detail) from error
    return completed.stdout.strip()
