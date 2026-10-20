from __future__ import annotations

import gc
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from review_sheep import (
    CachedGitCheckoutSource,
    Finding,
    GitCheckoutSource,
    PullRequestRevision,
    RepositoryMirrorLockTimeout,
    Review,
    ReviewCheckout,
    ReviewError,
    create_review_agent,
    git_changed_files,
    git_diff,
)
from review_sheep import checkout as checkout_module


class FakeRevisionSource:
    def __init__(self, revision: PullRequestRevision) -> None:
        self.revision = revision
        self.calls: list[tuple[str, int]] = []

    def get_pull_request_revision(
        self, *, repo: str, number: int
    ) -> PullRequestRevision:
        self.calls.append((repo, number))
        return self.revision


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "checkout"
    root.mkdir(parents=True)
    _git(root, "init")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "Review Sheep Tests")
    source = root / "example.py"
    source.write_text("value = 1\n")
    _git(root, "add", "example.py")
    _git(root, "commit", "-m", "base")
    base_sha = _git(root, "rev-parse", "HEAD")
    source.write_text("value = 2\n")
    (root / "new.py").write_text("enabled = True\n")
    _git(root, "add", "example.py", "new.py")
    _git(root, "commit", "-m", "head")
    return root, base_sha, _git(root, "rev-parse", "HEAD")


def _revision(base_sha: str, head_sha: str) -> PullRequestRevision:
    return PullRequestRevision(
        repo="acme/widgets",
        number=42,
        base_sha=base_sha,
        head_sha=head_sha,
        title="Bump value",
        body="Raises the value to 2.",
        head_ref="bump-value",
    )


def test_checkout_is_pinned_and_builds_changed_file_index_from_git(
    tmp_path: Path,
) -> None:
    root, base_sha, head_sha = _repository(tmp_path)
    revisions = FakeRevisionSource(_revision(base_sha, head_sha))

    checkout = GitCheckoutSource(revisions=revisions, root=root).prepare_checkout(
        repo="acme/widgets", number=42
    )

    assert checkout.root == root.resolve()
    assert checkout.base_sha == base_sha
    assert checkout.head_sha == head_sha
    assert checkout.title == "Bump value"
    assert checkout.body == "Raises the value to 2."
    assert checkout.head_ref == "bump-value"
    assert revisions.calls == [("acme/widgets", 42)]
    assert git_changed_files(checkout).splitlines() == [
        "M\texample.py",
        "A\tnew.py",
    ]
    assert "-value = 1" in git_diff(checkout, path="example.py")
    assert "+value = 2" in git_diff(checkout, path="example.py")
    assert "new.py" not in git_diff(checkout, path="example.py")


def test_checkout_rejects_a_head_other_than_the_pull_request_head(
    tmp_path: Path,
) -> None:
    root, base_sha, head_sha = _repository(tmp_path)
    source = GitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, "0" * len(head_sha))),
        root=root,
    )

    with pytest.raises(RuntimeError, match="checkout HEAD does not match"):
        source.prepare_checkout(repo="acme/widgets", number=42)


def test_checkout_rejects_uncommitted_files(tmp_path: Path) -> None:
    root, base_sha, head_sha = _repository(tmp_path)
    (root / "untracked.txt").write_text("not fixed\n")
    source = GitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        root=root,
    )

    with pytest.raises(RuntimeError, match="checkout must be clean"):
        source.prepare_checkout(repo="acme/widgets", number=42)


def test_checkout_rejects_a_missing_base_commit(tmp_path: Path) -> None:
    root, _, head_sha = _repository(tmp_path)
    source = GitCheckoutSource(
        revisions=FakeRevisionSource(_revision("f" * len(head_sha), head_sha)),
        root=root,
    )

    with pytest.raises(RuntimeError):
        source.prepare_checkout(repo="acme/widgets", number=42)


def test_checkout_rejects_a_missing_checkout_directory(tmp_path: Path) -> None:
    root = tmp_path / "missing-checkout"
    source = GitCheckoutSource(
        revisions=FakeRevisionSource(_revision("base", "head")),
        root=root,
    )

    with pytest.raises(RuntimeError, match="directory does not exist"):
        source.prepare_checkout(repo="acme/widgets", number=42)


def test_checkout_validation_reports_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="directory does not exist"):
        checkout_module._validate_checkout(
            root=tmp_path / "missing",
            revision=_revision("base", "head"),
        )


def test_checkout_path_must_be_the_worktree_root(tmp_path: Path) -> None:
    root, base_sha, head_sha = _repository(tmp_path)
    nested = root / "src"
    nested.mkdir()
    source = GitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        root=nested,
    )

    with pytest.raises(RuntimeError, match="must be the Git worktree root"):
        source.prepare_checkout(repo="acme/widgets", number=42)


def test_git_diff_reports_a_git_failure_as_an_exception(tmp_path: Path) -> None:
    root, _, _ = _repository(tmp_path)
    checkout = ReviewCheckout(
        repo="acme/widgets",
        pull_request_number=42,
        base_sha="0" * 40,
        head_sha="1" * 40,
        root=root,
    )

    with pytest.raises(RuntimeError, match="git command failed|fatal"):
        git_diff(checkout)


def test_cached_checkout_reuses_one_mirror_and_pins_isolated_worktrees(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    caplog.set_level("INFO")
    remote, base_sha, head_sha = _repository(tmp_path)
    revisions = FakeRevisionSource(_revision(base_sha, head_sha))
    source = CachedGitCheckoutSource(
        revisions=revisions,
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )

    first = source.prepare_checkout(repo="acme/widgets", number=42)
    second = source.prepare_checkout(repo="acme/widgets", number=42)

    assert first.root != second.root
    assert first.root.is_dir()
    assert second.root.is_dir()
    assert _git(first.root, "rev-parse", "HEAD") == head_sha
    assert _git(second.root, "rev-parse", "HEAD") == head_sha
    assert _git(first.root, "cat-file", "-e", f"{base_sha}^{{commit}}") == ""
    assert _git(second.root, "status", "--porcelain") == ""
    assert len(list((tmp_path / "cache").glob("*.git"))) == 1
    assert caplog.text.count("checkout.cache.mirror.initialize") == 1
    assert caplog.text.count("checkout.cache.mirror.reused") == 1
    assert caplog.text.count("checkout.cache.miss") == 1
    assert caplog.text.count("checkout.cache.hit") == 1
    assert caplog.text.count("checkout.cache.mirror.fetch.complete") == 2
    assert caplog.text.count("checkout.cache.worktree.created") == 2

    source.cleanup_checkout(first)
    source.cleanup_checkout(second)

    assert not first.root.exists()
    assert not second.root.exists()


def test_cached_checkout_ignores_cleanup_for_an_unknown_worktree(
    tmp_path: Path,
) -> None:
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision("base", "head")),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: "unused",
    )
    checkout = ReviewCheckout(
        repo="acme/widgets",
        pull_request_number=42,
        base_sha="base",
        head_sha="head",
        root=tmp_path / "unknown-worktree",
    )

    source.cleanup_checkout(checkout)

    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_cached_checkout_rejects_invalid_retention_configuration(
    value: float, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        CachedGitCheckoutSource(
            revisions=FakeRevisionSource(_revision("base", "head")),
            cache_root=tmp_path / "cache",
            remote_url=lambda _: "unused",
            worktree_retention=value,
        )


def test_cached_checkout_rejects_an_invalid_repository_name(tmp_path: Path) -> None:
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision("base", "head")),
        cache_root=tmp_path / "cache",
    )

    with pytest.raises(RuntimeError, match="owner/name form"):
        source.prepare_checkout(repo="not-a-repository", number=42)


def test_cached_checkout_requires_a_private_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir(mode=0o755)
    cache_root.chmod(0o755)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision("base", "head")),
        cache_root=cache_root,
        remote_url=lambda _: "unused",
    )

    with pytest.raises(RuntimeError, match="permissions 0700"):
        source.prepare_checkout(repo="acme/widgets", number=42)


def test_cached_checkout_requires_a_cache_root_owned_by_the_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir(mode=0o700)
    cache_root.chmod(0o700)
    actual_owner = cache_root.stat().st_uid
    monkeypatch.setattr(checkout_module.os, "getuid", lambda: actual_owner + 1)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision("base", "head")),
        cache_root=cache_root,
        remote_url=lambda _: "unused",
    )

    with pytest.raises(RuntimeError, match="owned by the review process"):
        source.prepare_checkout(repo="acme/widgets", number=42)


def test_cached_checkout_can_prune_a_worktree_removed_by_the_caller(
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )
    checkout = source.prepare_checkout(repo="acme/widgets", number=42)
    shutil.rmtree(checkout.root)

    source.cleanup_checkout(checkout)

    assert not checkout.root.exists()


def test_cached_checkout_rebuilds_a_mirror_file_and_keeps_working(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )
    first = source.prepare_checkout(repo="acme/widgets", number=42)
    source.cleanup_checkout(first)
    mirror = next((tmp_path / "cache").glob("*.git"))
    shutil.rmtree(mirror)
    mirror.write_text("not a bare repository")
    caplog.set_level("WARNING")

    rebuilt = source.prepare_checkout(repo="acme/widgets", number=42)

    assert rebuilt.head_sha == head_sha
    assert "quarantine_cleanup_failed" in caplog.text
    source.cleanup_checkout(rebuilt)


def test_cached_checkout_rebuilds_a_mirror_with_invalid_bare_metadata(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )
    first = source.prepare_checkout(repo="acme/widgets", number=42)
    source.cleanup_checkout(first)
    mirror = next((tmp_path / "cache").glob("*.git"))
    shutil.rmtree(mirror)
    mirror.mkdir()
    (mirror / "garbage").write_text("not git metadata")
    caplog.set_level("WARNING")

    rebuilt = source.prepare_checkout(repo="acme/widgets", number=42)

    assert rebuilt.head_sha == head_sha
    assert "mirror.rebuild.start" in caplog.text
    source.cleanup_checkout(rebuilt)


def test_cached_checkout_builds_the_default_github_repository_url() -> None:
    assert checkout_module._github_repository_url("acme/widgets") == (
        "https://github.com/acme/widgets.git"
    )


def test_cached_checkout_cleans_up_a_failed_bare_clone(tmp_path: Path) -> None:
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision("base", "head")),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(tmp_path / "missing-remote"),
    )

    with pytest.raises(RuntimeError):
        source.prepare_checkout(repo="acme/widgets", number=42)

    assert list((tmp_path / "cache").glob(".*.git-*")) == []


def test_cleanup_stale_keeps_a_recent_worktree(
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
        worktree_retention=10**9,
        mirror_retention=10**9,
    )
    checkout = source.prepare_checkout(repo="acme/widgets", number=42)
    cleaner = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
        worktree_retention=10**9,
        mirror_retention=10**9,
    )
    del source
    gc.collect()

    result = cleaner.cleanup_stale(now=time.time() + 1)

    assert result.removed_worktrees == 0
    assert result.errors == ()
    assert checkout.root.exists()


def test_cleanup_stale_reports_a_mirror_listing_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )
    checkout = source.prepare_checkout(repo="acme/widgets", number=42)
    source.cleanup_checkout(checkout)
    cleaner = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )

    def fail_to_list(_: Path) -> list[Path]:
        raise RuntimeError("worktree metadata unavailable")

    monkeypatch.setattr(cleaner, "_list_worktrees", fail_to_list)

    result = cleaner.cleanup_stale(now=time.time() + 1)

    assert result.errors == (
        "mirror " + str(next((tmp_path / "cache").glob("*.git"))) + ": RuntimeError",
    )


def test_file_lock_acquisition_returns_none_for_a_missing_parent(
    tmp_path: Path,
) -> None:
    assert (
        checkout_module._acquire_file_lock(
            tmp_path / "missing" / "lease", blocking=False
        )
        is None
    )


def test_mirror_lock_exit_is_safe_before_entering(tmp_path: Path) -> None:
    lock = checkout_module._RepositoryMirrorLock(
        tmp_path / "mirror.lock", timeout=0.1, label="test"
    )

    lock.__exit__(None, None, None)


def test_git_reports_a_missing_executable_as_a_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing_git(*_: object, **__: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr(checkout_module.subprocess, "run", missing_git)
    checkout = ReviewCheckout(
        repo="acme/widgets",
        pull_request_number=42,
        base_sha="base",
        head_sha="head",
        root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="git executable is not available"):
        git_diff(checkout)


def test_cached_checkout_coordinates_concurrent_same_repository_reviews(
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    revisions = FakeRevisionSource(_revision(base_sha, head_sha))
    sources = [
        CachedGitCheckoutSource(
            revisions=revisions,
            cache_root=tmp_path / "cache",
            remote_url=lambda _: str(remote),
        )
        for _ in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as workers:
        checkouts = list(
            workers.map(
                lambda source: source.prepare_checkout(repo="acme/widgets", number=42),
                sources,
            )
        )

    assert checkouts[0].root != checkouts[1].root
    assert all(
        _git(checkout.root, "rev-parse", "HEAD") == head_sha for checkout in checkouts
    )

    for source, checkout in zip(sources, checkouts, strict=True):
        source.cleanup_checkout(checkout)

    assert len(list((tmp_path / "cache").glob("*.git"))) == 1


def test_cached_checkout_reports_lock_contention_after_configured_timeout(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    caplog.set_level("INFO")
    remote, base_sha, head_sha = _repository(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def blocked_remote(_: str) -> str:
        entered.set()
        if not release.wait(timeout=2):
            raise RuntimeError("test lock holder was not released")
        return str(remote)

    revisions = FakeRevisionSource(_revision(base_sha, head_sha))
    holder = CachedGitCheckoutSource(
        revisions=revisions,
        cache_root=tmp_path / "cache",
        remote_url=blocked_remote,
    )
    contender = CachedGitCheckoutSource(
        revisions=revisions,
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
        lock_timeout=0.05,
    )

    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(
            holder.prepare_checkout,
            repo="acme/widgets",
            number=42,
        )
        assert entered.wait(timeout=2)
        try:
            with pytest.raises(RepositoryMirrorLockTimeout, match="timed out"):
                contender.prepare_checkout(repo="acme/widgets", number=42)
        finally:
            release.set()
        checkout = future.result()

    assert "checkout.cache.lock.contended" in caplog.text
    holder.cleanup_checkout(checkout)


def test_cached_checkout_releases_lock_after_failure_and_interruption(
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    revisions = FakeRevisionSource(_revision(base_sha, head_sha))

    def failing_remote(_: str) -> str:
        raise RuntimeError("remote unavailable")

    failing = CachedGitCheckoutSource(
        revisions=revisions,
        cache_root=tmp_path / "cache",
        remote_url=failing_remote,
    )
    working = CachedGitCheckoutSource(
        revisions=revisions,
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )

    with pytest.raises(RuntimeError, match="remote unavailable"):
        failing.prepare_checkout(repo="acme/widgets", number=42)
    checkout = working.prepare_checkout(repo="acme/widgets", number=42)
    working.cleanup_checkout(checkout)

    def interrupted_remote(_: str) -> str:
        raise KeyboardInterrupt

    interrupted = CachedGitCheckoutSource(
        revisions=revisions,
        cache_root=tmp_path / "interrupted-cache",
        remote_url=interrupted_remote,
    )
    after_interruption = CachedGitCheckoutSource(
        revisions=revisions,
        cache_root=tmp_path / "interrupted-cache",
        remote_url=lambda _: str(remote),
    )

    with pytest.raises(KeyboardInterrupt):
        interrupted.prepare_checkout(repo="acme/widgets", number=42)
    checkout = after_interruption.prepare_checkout(repo="acme/widgets", number=42)
    after_interruption.cleanup_checkout(checkout)


def test_cached_checkout_uses_independent_locks_for_different_repositories(
    tmp_path: Path,
) -> None:
    first_remote, first_base, first_head = _repository(tmp_path / "first")
    second_remote, second_base, second_head = _repository(tmp_path / "second")
    revisions_by_repo = {
        "acme/widgets": _revision(first_base, first_head),
        "other/gadgets": PullRequestRevision(
            repo="other/gadgets",
            number=42,
            base_sha=second_base,
            head_sha=second_head,
        ),
    }
    barrier = threading.Barrier(2)

    class MultiRevisionSource:
        def get_pull_request_revision(
            self, *, repo: str, number: int
        ) -> PullRequestRevision:
            return revisions_by_repo[repo]

    def remote_url(repo: str) -> str:
        barrier.wait(timeout=2)
        return {
            "acme/widgets": str(first_remote),
            "other/gadgets": str(second_remote),
        }[repo]

    source = CachedGitCheckoutSource(
        revisions=MultiRevisionSource(),
        cache_root=tmp_path / "cache",
        remote_url=remote_url,
    )

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(source.prepare_checkout, repo=repo, number=42)
            for repo in ("acme/widgets", "other/gadgets")
        ]
        checkouts = [future.result() for future in futures]

    assert {checkout.head_sha for checkout in checkouts} == {first_head, second_head}
    for checkout in checkouts:
        source.cleanup_checkout(checkout)


def test_cached_checkout_keeps_different_repositories_in_different_mirrors(
    tmp_path: Path,
) -> None:
    first_remote, first_base, first_head = _repository(tmp_path / "first")
    second_remote, second_base, second_head = _repository(tmp_path / "second")
    revisions = FakeRevisionSource(_revision(first_base, first_head))
    revisions_by_repo = {
        "acme/widgets": _revision(first_base, first_head),
        "other/gadgets": PullRequestRevision(
            repo="other/gadgets",
            number=42,
            base_sha=second_base,
            head_sha=second_head,
        ),
    }

    class MultiRevisionSource(FakeRevisionSource):
        def get_pull_request_revision(
            self, *, repo: str, number: int
        ) -> PullRequestRevision:
            self.calls.append((repo, number))
            return revisions_by_repo[repo]

    source = CachedGitCheckoutSource(
        revisions=MultiRevisionSource(revisions.revision),
        cache_root=tmp_path / "cache",
        remote_url=lambda repo: str(
            {"acme/widgets": first_remote, "other/gadgets": second_remote}[repo]
        ),
    )

    first = source.prepare_checkout(repo="acme/widgets", number=42)
    second = source.prepare_checkout(repo="other/gadgets", number=42)

    assert first.root != second.root
    assert _git(first.root, "rev-parse", "HEAD") == first_head
    assert _git(second.root, "rev-parse", "HEAD") == second_head
    assert len(list((tmp_path / "cache").glob("*.git"))) == 2

    source.cleanup_checkout(first)
    source.cleanup_checkout(second)


def test_cached_checkout_rejects_credentials_in_repository_url(tmp_path: Path) -> None:
    _, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: "https://token@example.com/acme/widgets.git",
    )

    with pytest.raises(RuntimeError, match="must not embed credentials"):
        source.prepare_checkout(repo="acme/widgets", number=42)


@pytest.mark.parametrize("fails", [False, True])
def test_cached_checkout_is_cleaned_by_review_graph_on_success_or_failure(
    fails: bool,
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )
    reviewed_roots: list[Path] = []

    class Runner:
        def run(self, checkout: ReviewCheckout) -> list[Finding]:
            reviewed_roots.append(checkout.root)
            if fails:
                raise RuntimeError("runner failed")
            return []

    result = create_review_agent(source=source, runner=Runner()).review(
        repo="acme/widgets", number=42
    )

    assert isinstance(result, ReviewError if fails else Review)
    assert len(reviewed_roots) == 1
    assert not reviewed_roots[0].exists()
    assert len(list((tmp_path / "cache").glob("*.git"))) == 1


def test_cached_checkout_retries_after_failed_mirror_initialization(
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    attempts = 0

    def flaky_remote(_: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("remote unavailable")
        return str(remote)

    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=flaky_remote,
    )

    with pytest.raises(RuntimeError, match="remote unavailable"):
        source.prepare_checkout(repo="acme/widgets", number=42)
    assert list((tmp_path / "cache").glob("*.git")) == []

    checkout = source.prepare_checkout(repo="acme/widgets", number=42)
    source.cleanup_checkout(checkout)
    assert attempts == 2


def test_cached_checkout_rebuilds_a_corrupt_mirror_safely(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )
    first = source.prepare_checkout(repo="acme/widgets", number=42)
    source.cleanup_checkout(first)
    mirror = next((tmp_path / "cache").glob("*.git"))
    (mirror / "config").unlink()
    caplog.set_level("WARNING")

    rebuilt = source.prepare_checkout(repo="acme/widgets", number=42)

    assert rebuilt.head_sha == head_sha
    assert "checkout.cache.mirror.rebuild.start" in caplog.text
    source.cleanup_checkout(rebuilt)


def test_cached_checkout_rebuilds_after_failed_revision_fetch(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    valid_revisions = FakeRevisionSource(_revision(base_sha, head_sha))
    source = CachedGitCheckoutSource(
        revisions=valid_revisions,
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )
    first = source.prepare_checkout(repo="acme/widgets", number=42)
    source.cleanup_checkout(first)
    invalid = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, "0" * len(head_sha))),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )

    with pytest.raises(RuntimeError):
        invalid.prepare_checkout(repo="acme/widgets", number=42)

    caplog.set_level("WARNING")
    recovered = source.prepare_checkout(repo="acme/widgets", number=42)
    assert recovered.head_sha == head_sha
    assert "checkout.cache.fetch.recovery" in caplog.text
    source.cleanup_checkout(recovered)


def test_cleanup_stale_skips_active_and_removes_released_worktrees(
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
        worktree_retention=0,
        mirror_retention=10**9,
    )
    active = source.prepare_checkout(repo="acme/widgets", number=42)
    cleaner = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
        worktree_retention=0,
        mirror_retention=10**9,
    )

    result = cleaner.cleanup_stale(now=time.time() + 1)

    assert result.removed_worktrees == 0
    assert result.skipped_active_worktrees == 1
    assert active.root.exists()
    source.cleanup_checkout(active)

    stale = source.prepare_checkout(repo="acme/widgets", number=42)
    del source
    gc.collect()
    result = cleaner.cleanup_stale(now=time.time() + 1)

    assert result.removed_worktrees == 1
    assert not stale.root.exists()


def test_cleanup_stale_removes_old_mirror_after_worktrees_are_gone(
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )
    checkout = source.prepare_checkout(repo="acme/widgets", number=42)
    source.cleanup_checkout(checkout)
    cleaner = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
        worktree_retention=0,
        mirror_retention=0,
    )

    result = cleaner.cleanup_stale(now=time.time() + 1)

    assert result.removed_mirrors == 1
    assert list((tmp_path / "cache").glob("*.git")) == []


def test_cleanup_stale_releases_lease_after_interrupted_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    source = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
        worktree_retention=0,
        mirror_retention=10**9,
    )
    checkout = source.prepare_checkout(repo="acme/widgets", number=42)
    original_git = checkout_module._git

    def interrupted_git(*_: object) -> str:
        raise KeyboardInterrupt("cleanup interrupted")

    monkeypatch.setattr(checkout_module, "_git", interrupted_git)
    with pytest.raises(KeyboardInterrupt, match="cleanup interrupted"):
        source.cleanup_checkout(checkout)
    monkeypatch.setattr(checkout_module, "_git", original_git)

    cleaner = CachedGitCheckoutSource(
        revisions=FakeRevisionSource(_revision(base_sha, head_sha)),
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
        worktree_retention=0,
        mirror_retention=10**9,
    )
    result = cleaner.cleanup_stale(now=time.time() + 1)

    assert result.removed_worktrees == 1
    assert not checkout.root.exists()
