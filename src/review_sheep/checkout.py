"""Validate the fixed local Git checkout used by Review."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
from pathlib import Path
from typing import Protocol, Self, TextIO
from urllib.parse import urlparse
from uuid import uuid4

from review_sheep.domain import PullRequestRevision, ReviewCheckout

logger = logging.getLogger(__name__)


class RepositoryMirrorLockTimeout(RuntimeError):
    """A repository mirror lock was not acquired before its configured timeout."""


@dataclass(frozen=True)
class RepositoryCacheCleanupResult:
    """Outcome of one stale repository-cache cleanup pass."""

    removed_worktrees: int = 0
    removed_mirrors: int = 0
    skipped_active_worktrees: int = 0
    errors: tuple[str, ...] = ()


@dataclass
class _ActiveWorktree:
    mirror: Path
    lease_path: Path
    lease: TextIO


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


class _RepositoryMirrorLock:
    """Coordinate operations that mutate one repository mirror."""

    def __init__(self, path: Path, *, timeout: float, label: str) -> None:
        self._path = path
        self._timeout = timeout
        self._label = label
        self._handle: TextIO | None = None

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+")
        self._handle = handle
        deadline = time.monotonic() + self._timeout
        contended = False
        try:
            while True:
                try:
                    flock(handle.fileno(), LOCK_EX | LOCK_NB)
                    if contended:
                        logger.info(
                            "checkout.cache.lock.acquired_after_contention label=%s",
                            self._label,
                        )
                    return self
                except BlockingIOError:
                    if not contended:
                        contended = True
                        logger.info(
                            "checkout.cache.lock.contended label=%s timeout=%.3f",
                            self._label,
                            self._timeout,
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RepositoryMirrorLockTimeout(
                            "timed out acquiring repository mirror lock for "
                            f"{self._label} after {self._timeout:.3f}s"
                        )
                    time.sleep(min(0.05, remaining))
        except BaseException:
            self._handle = None
            handle.close()
            raise

    def __exit__(self, *_: object) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            flock(handle.fileno(), LOCK_UN)
        finally:
            handle.close()


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
        lock_timeout: float = 30.0,
        worktree_retention: float = 24 * 60 * 60,
        mirror_retention: float = 30 * 24 * 60 * 60,
    ) -> None:
        _validate_non_negative_finite(lock_timeout, "lock_timeout")
        _validate_non_negative_finite(worktree_retention, "worktree_retention")
        _validate_non_negative_finite(mirror_retention, "mirror_retention")
        self._revisions = revisions
        self._cache_root = Path(cache_root).expanduser().resolve()
        self._worktree_root = (
            Path(worktree_root).expanduser().resolve()
            if worktree_root is not None
            else self._cache_root / "worktrees"
        )
        self._remote_url = remote_url or _github_repository_url
        self._lock_timeout = lock_timeout
        self._worktree_retention = worktree_retention
        self._mirror_retention = mirror_retention
        self._active_worktrees: dict[Path, _ActiveWorktree] = {}

    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout:
        """Fetch immutable revisions and create one clean disposable worktree."""
        self._ensure_cache_root()
        revision = self._revisions.get_pull_request_revision(repo=repo, number=number)
        mirror = self._mirror_path(repo)
        logger.info(
            "checkout.cache.prepare.start repo=%s pr=%d mirror=%s",
            repo,
            number,
            mirror,
        )
        with _RepositoryMirrorLock(
            self._lock_path(repo),
            timeout=self._lock_timeout,
            label=repo,
        ):
            remote = self._remote_url(repo)
            _reject_embedded_credentials(remote)
            mirror_reused = self._ensure_mirror(
                mirror=mirror,
                remote=remote,
                repo=repo,
            )
            logger.info(
                "checkout.cache.%s repo=%s mirror=%s",
                "hit" if mirror_reused else "miss",
                repo,
                mirror,
            )
            try:
                self._fetch_revision(
                    mirror=mirror,
                    revision=revision,
                    number=number,
                )
            except RuntimeError as error:
                logger.warning(
                    "checkout.cache.fetch.recovery repo=%s error_type=%s",
                    repo,
                    type(error).__name__,
                )
                self._rebuild_mirror(
                    mirror=mirror,
                    remote=remote,
                    repo=repo,
                    reason=error,
                )
                self._fetch_revision(
                    mirror=mirror,
                    revision=revision,
                    number=number,
                )
            worktree = self._new_worktree_path(repo)
            lease_path = self._lease_path(worktree)
            lease: TextIO | None = None
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
                lease = _acquire_file_lock(lease_path, blocking=False)
                if lease is None:
                    raise RuntimeError(f"worktree lease is already active: {worktree}")
                checkout = _validate_checkout(
                    root=worktree,
                    revision=revision,
                )
                logger.info(
                    "checkout.cache.worktree.created repo=%s pr=%d root=%s head=%s",
                    repo,
                    number,
                    worktree,
                    checkout.head_sha,
                )
            except BaseException:
                if lease is not None:
                    _release_file_lock(lease)
                try:
                    self._remove_worktree(mirror=mirror, root=worktree)
                except Exception as error:  # noqa: BLE001 - cleanup must not mask checkout error
                    logger.error(
                        "checkout.cache.prepare.cleanup_failed root=%s error_type=%s",
                        worktree,
                        type(error).__name__,
                    )
                raise
            self._active_worktrees[checkout.root] = _ActiveWorktree(
                mirror=mirror,
                lease_path=lease_path,
                lease=lease,
            )
            self._touch_mirror_activity(mirror)
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
        active = self._active_worktrees.get(root)
        if active is None:
            return
        try:
            with _RepositoryMirrorLock(
                active.mirror.with_suffix(".lock"),
                timeout=self._lock_timeout,
                label=str(active.mirror),
            ):
                self._remove_worktree(mirror=active.mirror, root=root)
        except BaseException as error:
            logger.error(
                "checkout.cache.cleanup.failed root=%s error_type=%s",
                root,
                type(error).__name__,
            )
            raise
        finally:
            _release_file_lock(active.lease)
            if not root.exists():
                self._active_worktrees.pop(root, None)
                active.lease_path.unlink(missing_ok=True)
        logger.info("checkout.cache.cleanup.complete root=%s", root)

    def cleanup_stale(
        self, *, now: float | None = None
    ) -> RepositoryCacheCleanupResult:
        """Remove stale worktrees and mirrors while preserving active resources.

        Worktrees remain eligible only after ``worktree_retention`` seconds and
        mirrors only after ``mirror_retention`` seconds without use. Active
        worktrees hold a lease lock, so a cleanup pass from another process
        skips them safely.
        """
        self._ensure_cache_root()
        current_time = time.time() if now is None else now
        removed_worktrees = 0
        removed_mirrors = 0
        skipped_active = 0
        errors: list[str] = []
        for mirror in sorted(self._cache_root.glob("*.git")):
            try:
                with _RepositoryMirrorLock(
                    mirror.with_suffix(".lock"),
                    timeout=self._lock_timeout,
                    label=str(mirror),
                ):
                    worktrees = self._list_worktrees(mirror)
                    for root in worktrees:
                        if not root.is_relative_to(self._worktree_root):
                            continue
                        lease_path = self._lease_path(root)
                        last_used = _resource_mtime(
                            lease_path=lease_path,
                            resource=root,
                        )
                        lease = _acquire_file_lock(lease_path, blocking=False)
                        if lease is None:
                            skipped_active += 1
                            logger.info(
                                "checkout.cache.cleanup.skip_active root=%s", root
                            )
                            continue
                        try:
                            if not _is_stale(
                                last_used=last_used,
                                now=current_time,
                                retention=self._worktree_retention,
                            ):
                                continue
                            self._remove_worktree(mirror=mirror, root=root)
                            removed_worktrees += 1
                            logger.info(
                                "checkout.cache.cleanup.worktree_removed root=%s",
                                root,
                            )
                        except Exception as error:  # noqa: BLE001 - report and continue cleanup
                            message = f"worktree {root}: {type(error).__name__}"
                            errors.append(message)
                            logger.error(
                                "checkout.cache.cleanup.worktree_failed "
                                "root=%s error_type=%s",
                                root,
                                type(error).__name__,
                            )
                        finally:
                            _release_file_lock(lease)
                            if not root.exists():
                                lease_path.unlink(missing_ok=True)

                    if not worktrees and _is_stale(
                        last_used=_resource_mtime(
                            lease_path=self._last_used_path(mirror),
                            resource=mirror,
                        ),
                        now=current_time,
                        retention=self._mirror_retention,
                    ):
                        shutil.rmtree(mirror)
                        self._last_used_path(mirror).unlink(missing_ok=True)
                        removed_mirrors += 1
                        logger.info(
                            "checkout.cache.cleanup.mirror_removed mirror=%s", mirror
                        )
            except Exception as error:  # noqa: BLE001 - report and continue cleanup
                message = f"mirror {mirror}: {type(error).__name__}"
                errors.append(message)
                logger.error(
                    "checkout.cache.cleanup.mirror_failed mirror=%s error_type=%s",
                    mirror,
                    type(error).__name__,
                )
        return RepositoryCacheCleanupResult(
            removed_worktrees=removed_worktrees,
            removed_mirrors=removed_mirrors,
            skipped_active_worktrees=skipped_active,
            errors=tuple(errors),
        )

    def _mirror_path(self, repo: str) -> Path:
        return self._cache_root / f"{_repository_cache_key(repo)}.git"

    def _ensure_cache_root(self) -> None:
        self._cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        permissions = self._cache_root.stat().st_mode & 0o777
        if permissions & 0o077:
            raise RuntimeError(
                "repository cache root must be private with permissions 0700"
            )
        if hasattr(os, "getuid") and self._cache_root.stat().st_uid != os.getuid():
            raise RuntimeError(
                "repository cache root must be owned by the review process"
            )

    def _lock_path(self, repo: str) -> Path:
        return self._cache_root / f"{_repository_cache_key(repo)}.lock"

    def _lease_path(self, root: Path) -> Path:
        return self._worktree_root / f".{root.name}.lease"

    def _last_used_path(self, mirror: Path) -> Path:
        return mirror.with_suffix(".last-used")

    def _touch_mirror_activity(self, mirror: Path) -> None:
        self._last_used_path(mirror).touch(exist_ok=True)

    def _list_worktrees(self, mirror: Path) -> list[Path]:
        output = _git(
            mirror.parent,
            "--git-dir",
            str(mirror),
            "worktree",
            "list",
            "--porcelain",
        )
        paths: list[Path] = []
        for line in output.splitlines():
            if not line.startswith("worktree "):
                continue
            root = Path(line.removeprefix("worktree ")).resolve()
            if root != mirror.resolve():
                paths.append(root)
        return paths

    def _has_active_worktree(self, *, mirror: Path, repo: str) -> bool:
        if any(active.mirror == mirror for active in self._active_worktrees.values()):
            return True
        pattern = f".{_repository_cache_key(repo)}-*.lease"
        for lease_path in self._worktree_root.glob(pattern):
            lease = _acquire_file_lock(lease_path, blocking=False)
            if lease is None:
                return True
            _release_file_lock(lease)
        return False

    def _rebuild_mirror(
        self,
        *,
        mirror: Path,
        remote: str,
        repo: str,
        reason: Exception,
    ) -> None:
        if self._has_active_worktree(mirror=mirror, repo=repo):
            raise RuntimeError(
                f"cannot rebuild repository mirror while an active worktree "
                f"exists for {repo}: {reason}"
            ) from reason
        logger.warning(
            "checkout.cache.mirror.rebuild.start repo=%s mirror=%s reason_type=%s",
            repo,
            mirror,
            type(reason).__name__,
        )
        temporary_mirror = self._clone_to_temporary(mirror=mirror, remote=remote)
        quarantine = mirror.parent / f".{mirror.name}.corrupt-{uuid4().hex}"
        try:
            if mirror.exists():
                os.replace(mirror, quarantine)
            os.replace(temporary_mirror, mirror)
        except BaseException:
            if temporary_mirror.exists():
                shutil.rmtree(temporary_mirror, ignore_errors=True)
            if quarantine.exists() and not mirror.exists():
                os.replace(quarantine, mirror)
            raise
        if quarantine.exists():
            try:
                shutil.rmtree(quarantine)
            except OSError:
                logger.exception(
                    "checkout.cache.mirror.quarantine_cleanup_failed path=%s",
                    quarantine,
                )
        logger.info("checkout.cache.mirror.rebuild.complete mirror=%s", mirror)

    def _remove_worktree(self, *, mirror: Path, root: Path) -> None:
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
        else:
            _git(
                mirror.parent,
                "--git-dir",
                str(mirror),
                "worktree",
                "prune",
                "--expire",
                "now",
            )
        if root.exists():
            shutil.rmtree(root)

    def _ensure_mirror(self, *, mirror: Path, remote: str, repo: str) -> bool:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        if mirror.exists():
            if not mirror.is_dir():
                corruption = RuntimeError(
                    f"repository mirror path is not a directory: {mirror}"
                )
                self._rebuild_mirror(
                    mirror=mirror,
                    remote=remote,
                    repo=repo,
                    reason=corruption,
                )
                return False
            try:
                is_bare = _git(
                    mirror.parent,
                    "--git-dir",
                    str(mirror),
                    "rev-parse",
                    "--is-bare-repository",
                )
            except RuntimeError as error:
                self._rebuild_mirror(
                    mirror=mirror,
                    remote=remote,
                    repo=repo,
                    reason=error,
                )
                return False
            if is_bare != "true":
                corruption = RuntimeError(f"repository mirror is not bare: {mirror}")
                self._rebuild_mirror(
                    mirror=mirror,
                    remote=remote,
                    repo=repo,
                    reason=corruption,
                )
                return False
            logger.info("checkout.cache.mirror.reused mirror=%s", mirror)
            return True
        logger.info("checkout.cache.mirror.initialize mirror=%s", mirror)
        temporary_mirror = self._clone_to_temporary(mirror=mirror, remote=remote)
        try:
            os.replace(temporary_mirror, mirror)
        except BaseException:
            shutil.rmtree(temporary_mirror, ignore_errors=True)
            raise
        return False

    def _clone_to_temporary(self, *, mirror: Path, remote: str) -> Path:
        temporary_mirror = Path(
            tempfile.mkdtemp(prefix=f".{mirror.name}-", dir=mirror.parent)
        )
        try:
            _git(mirror.parent, "clone", "--bare", remote, str(temporary_mirror))
        except BaseException:
            shutil.rmtree(temporary_mirror, ignore_errors=True)
            raise
        return temporary_mirror

    def _fetch_revision(
        self,
        *,
        mirror: Path,
        revision: PullRequestRevision,
        number: int,
    ) -> None:
        logger.info(
            "checkout.cache.mirror.fetch.start repo=%s pr=%d",
            revision.repo,
            number,
        )
        try:
            _git(
                mirror.parent,
                "--git-dir",
                str(mirror),
                "fetch",
                "--atomic",
                "--prune",
                "origin",
                revision.base_sha,
                revision.head_sha,
            )
        except RuntimeError as error:
            # GitHub exposes a pull request head ref even when its commit is not
            # reachable through the repository's ordinary advertised branches.
            logger.warning(
                "checkout.cache.fetch.shas_failed repo=%s pr=%d error_type=%s",
                revision.repo,
                number,
                type(error).__name__,
            )
            _git(
                mirror.parent,
                "--git-dir",
                str(mirror),
                "fetch",
                "--atomic",
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
        logger.info(
            "checkout.cache.mirror.fetch.complete repo=%s pr=%d",
            revision.repo,
            number,
        )

    def _new_worktree_path(self, repo: str) -> Path:
        self._worktree_root.mkdir(parents=True, exist_ok=True)
        return self._worktree_root / f"{_repository_cache_key(repo)}-{uuid4().hex}"


def _acquire_file_lock(path: Path, *, blocking: bool) -> TextIO | None:
    try:
        handle = path.open("a+")
    except FileNotFoundError:
        return None
    try:
        operation = LOCK_EX if blocking else LOCK_EX | LOCK_NB
        flock(handle.fileno(), operation)
    except BlockingIOError:
        handle.close()
        return None
    except BaseException:
        handle.close()
        raise
    return handle


def _release_file_lock(handle: TextIO) -> None:
    try:
        flock(handle.fileno(), LOCK_UN)
    finally:
        handle.close()


def _resource_mtime(*, lease_path: Path, resource: Path) -> float:
    timestamp_path = lease_path if lease_path.exists() else resource
    return timestamp_path.stat().st_mtime


def _is_stale(*, last_used: float, now: float, retention: float) -> bool:
    return now - last_used >= retention


def _validate_non_negative_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


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
