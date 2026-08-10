from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from review_sheep import (
    GitCheckoutSource,
    PullRequestRevision,
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
    root.mkdir()
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
