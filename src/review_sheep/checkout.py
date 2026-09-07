"""Validate the fixed local Git checkout used by Review."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse
from uuid import uuid4

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
        checkout = _validate_checkout(root=self._root, revision=revision)
        logger.info(
            "checkout.prepare.revisions expected_head=%s actual_head=%s base=%s",
            revision.head_sha,
            checkout.head_sha,
            revision.base_sha,
        )
        logger.info("checkout.prepare.complete clean=true")
        return checkout


class CachedGitCheckoutSource:
    """Prepare isolated fixed-SHA worktrees backed by a persistent Git mirror.

    The mirror stores only the repository URL and Git objects. Authentication is
    delegated to Git's credential helpers or environment, so callers must not
    provide a URL containing credentials. Each prepared worktree is tracked by
    this source and can be removed with :meth:`cleanup_checkout`.
    """

    def __init__(
        self,
        *,
        revisions: PullRequestRevisionSource,
        cache_root: str | Path,
        worktree_root: str | Path | None = None,
        remote_url: Callable[[str], str] | None = None,
    ) -> None:
        self._revisions = revisions
        self._cache_root = Path(cache_root).expanduser().resolve()
        self._worktree_root = (
            Path(worktree_root).expanduser().resolve()
            if worktree_root is not None
            else self._cache_root / "worktrees"
        )
        self._remote_url = remote_url or _github_repository_url
        self._active_worktrees: dict[Path, Path] = {}

    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout:
        """Fetch immutable revisions and create one clean disposable worktree."""
        revision = self._revisions.get_pull_request_revision(repo=repo, number=number)
        remote = self._remote_url(repo)
        _reject_embedded_credentials(remote)
        mirror = self._mirror_path(repo)
        logger.info(
            "checkout.cache.prepare.start repo=%s pr=%d mirror=%s",
            repo,
            number,
            mirror,
        )
        mirror_reused = self._ensure_mirror(mirror=mirror, remote=remote)
        self._fetch_revision(
            mirror=mirror,
            revision=revision,
            number=number,
        )
        worktree = self._new_worktree_path(repo)
        try:
            _git(
                mirror.parent,
                "--git-dir",
                str(mirror),
                "worktree",
                "add",
                "--detach",
                str(worktree),
                revision.head_sha,
            )
            checkout = _validate_checkout(
                root=worktree,
                revision=revision,
            )
        except Exception:
            if worktree.exists():
                _git(
                    mirror.parent,
                    "--git-dir",
                    str(mirror),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree),
                )
                if worktree.exists():
                    shutil.rmtree(worktree)
            raise
        self._active_worktrees[checkout.root] = mirror
        logger.info(
            "checkout.cache.prepare.complete repo=%s pr=%d mirror_reused=%s root=%s",
            repo,
            number,
            mirror_reused,
            checkout.root,
        )
        return checkout

    def cleanup_checkout(self, checkout: ReviewCheckout) -> None:
        """Remove a worktree created by this source while retaining its mirror."""
        root = checkout.root.resolve()
        mirror = self._active_worktrees.get(root)
        if mirror is None:
            return
        try:
            if root.exists():
                _git(
                    mirror.parent,
                    "--git-dir",
                    str(mirror),
                    "worktree",
                    "remove",
                    "--force",
                    str(root),
                )
            if root.exists():
                shutil.rmtree(root)
        except Exception:
            logger.exception("checkout.cache.cleanup.failed root=%s", root)
            raise
        finally:
            if not root.exists():
                self._active_worktrees.pop(root, None)
        logger.info("checkout.cache.cleanup.complete root=%s", root)

    def _mirror_path(self, repo: str) -> Path:
        return self._cache_root / f"{_repository_cache_key(repo)}.git"

    def _ensure_mirror(self, *, mirror: Path, remote: str) -> bool:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        if mirror.exists():
            if not mirror.is_dir():
                raise RuntimeError(
                    f"repository mirror path is not a directory: {mirror}"
                )
            if (
                _git(
                    mirror.parent,
                    "--git-dir",
                    str(mirror),
                    "rev-parse",
                    "--is-bare-repository",
                )
                != "true"
            ):
                raise RuntimeError(f"repository mirror is not bare: {mirror}")
            logger.info("checkout.cache.mirror.reused mirror=%s", mirror)
            return True
        logger.info("checkout.cache.mirror.initialize mirror=%s", mirror)
        _git(mirror.parent, "clone", "--bare", remote, str(mirror))
        return False

    def _fetch_revision(
        self,
        *,
        mirror: Path,
        revision: PullRequestRevision,
        number: int,
    ) -> None:
        try:
            _git(
                mirror.parent,
                "--git-dir",
                str(mirror),
                "fetch",
                "--prune",
                "origin",
                revision.base_sha,
                revision.head_sha,
            )
        except RuntimeError as error:
            # GitHub exposes a pull request head ref even when its commit is not
            # reachable through the repository's ordinary advertised branches.
            logger.warning(
                "checkout.cache.fetch.shas_failed repo=%s pr=%d error=%s",
                revision.repo,
                number,
                error,
            )
            _git(
                mirror.parent,
                "--git-dir",
                str(mirror),
                "fetch",
                "--prune",
                "origin",
                f"+refs/pull/{number}/head:refs/remotes/review-sheep/pr-{number}",
                revision.base_sha,
            )
        _git(
            mirror.parent,
            "--git-dir",
            str(mirror),
            "cat-file",
            "-e",
            f"{revision.base_sha}^{{commit}}",
        )
        _git(
            mirror.parent,
            "--git-dir",
            str(mirror),
            "cat-file",
            "-e",
            f"{revision.head_sha}^{{commit}}",
        )

    def _new_worktree_path(self, repo: str) -> Path:
        self._worktree_root.mkdir(parents=True, exist_ok=True)
        return self._worktree_root / f"{_repository_cache_key(repo)}-{uuid4().hex}"


def _validate_checkout(*, root: Path, revision: PullRequestRevision) -> ReviewCheckout:
    if not root.is_dir():
        raise RuntimeError(f"review checkout directory does not exist: {root}")
    resolved_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if resolved_root != root.resolve():
        raise RuntimeError(
            "review checkout path must be the Git worktree root: "
            f"expected {resolved_root}, received {root.resolve()}"
        )
    head_sha = _git(resolved_root, "rev-parse", "HEAD")
    if head_sha != revision.head_sha:
        raise RuntimeError(
            "checkout HEAD does not match pull request head: "
            f"expected {revision.head_sha}, found {head_sha}"
        )
    _git(resolved_root, "cat-file", "-e", f"{revision.base_sha}^{{commit}}")
    dirty = _git(resolved_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError("review checkout must be clean")
    return ReviewCheckout(
        repo=revision.repo,
        pull_request_number=revision.number,
        base_sha=revision.base_sha,
        head_sha=revision.head_sha,
        root=resolved_root,
    )


def _repository_cache_key(repo: str) -> str:
    digest = hashlib.sha256(repo.encode("utf-8")).hexdigest()[:12]
    slug = repo.replace("/", "-").replace("\\", "-")
    return f"{slug}-{digest}"


def _github_repository_url(repo: str) -> str:
    if repo.count("/") != 1 or any(part in repo for part in ("..", "\\")):
        raise RuntimeError(f"repository must be in owner/name form: {repo}")
    return f"https://github.com/{repo}.git"


def _reject_embedded_credentials(remote: str) -> None:
    parsed = urlparse(remote)
    if parsed.scheme in {"http", "https"} and (
        parsed.username or parsed.password or parsed.query or parsed.fragment
    ):
        raise RuntimeError("repository URL must not embed credentials")


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
