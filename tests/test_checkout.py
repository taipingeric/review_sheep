from __future__ import annotations

import subprocess
import threading
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


def test_cached_checkout_reuses_one_mirror_and_pins_isolated_worktrees(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    remote, base_sha, head_sha = _repository(tmp_path)
    revisions = FakeRevisionSource(_revision(base_sha, head_sha))
    source = CachedGitCheckoutSource(
        revisions=revisions,
        cache_root=tmp_path / "cache",
        remote_url=lambda _: str(remote),
    )
    caplog.set_level("INFO")

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

    source.cleanup_checkout(first)
    source.cleanup_checkout(second)

    assert not first.root.exists()
    assert not second.root.exists()


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
    tmp_path: Path,
) -> None:
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
