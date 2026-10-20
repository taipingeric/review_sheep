from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from review_sheep import (
    GitHubPullRequestReader,
    PullRequestRevision,
    PullRequestSnapshot,
    SnapshotFile,
)


class FakeUser(BaseModel):
    login: str


class FakePullRequest(BaseModel):
    number: int
    title: str
    html_url: str
    state: str = "open"
    user: FakeUser | None = None
    created_at: Any = None
    updated_at: Any = None


class FakeRepository:
    def __init__(
        self, pulls: list[FakePullRequest], pull_details: Any | None = None
    ) -> None:
        self.pulls = pulls
        self.pull_details = pull_details
        self.queries: list[dict[str, str]] = []
        self.requested_pulls: list[int] = []

    def get_pulls(self, *, state: str, sort: str, direction: str) -> list[Any]:
        self.queries.append({"state": state, "sort": sort, "direction": direction})
        return self.pulls

    def get_pull(self, number: int) -> Any:
        self.requested_pulls.append(number)
        assert self.pull_details is not None
        assert number == self.pull_details.number
        return self.pull_details


class FakeGithubClient:
    def __init__(self, repository: FakeRepository) -> None:
        self.repository = repository
        self.requested_repositories: list[str] = []

    def get_repo(self, repo: str) -> FakeRepository:
        self.requested_repositories.append(repo)
        return self.repository


def test_reader_uses_default_repo_and_marks_a_limited_list_as_truncated() -> None:
    repository = FakeRepository(
        [
            FakePullRequest(
                number=42,
                title="Keep the flock together",
                html_url="https://github.com/acme/widgets/pull/42",
                user=FakeUser(login="alice"),
            ),
            FakePullRequest(
                number=41,
                title="Count every sheep",
                html_url="https://github.com/acme/widgets/pull/41",
                user=FakeUser(login="bob"),
            ),
        ]
    )
    client = FakeGithubClient(repository)
    reader = GitHubPullRequestReader(client=client, default_repo="acme/widgets")

    result = reader.list_pull_requests(state="open", limit=1, repo="")

    assert client.requested_repositories == ["acme/widgets"]
    assert repository.queries == [
        {"state": "open", "sort": "updated", "direction": "desc"}
    ]
    assert result == {
        "repo": "acme/widgets",
        "state": "open",
        "count": 1,
        "truncated": True,
        "pull_requests": [
            {
                "number": 42,
                "title": "Keep the flock together",
                "state": "open",
                "author": "alice",
                "created_at": None,
                "updated_at": None,
                "url": "https://github.com/acme/widgets/pull/42",
            }
        ],
    }


def test_reader_requires_a_repository_when_no_default_is_configured() -> None:
    reader = GitHubPullRequestReader(client=object())

    with pytest.raises(RuntimeError, match="no repository given"):
        reader.list_pull_requests(state="open", limit=1, repo="")


def test_reader_propagates_github_api_failures() -> None:
    class FailingClient:
        def get_repo(self, repo: str) -> Any:
            raise RuntimeError(f"GitHub API unavailable for {repo}")

    reader = GitHubPullRequestReader(
        client=FailingClient(), default_repo="acme/widgets"
    )

    with pytest.raises(RuntimeError, match="GitHub API unavailable"):
        reader.get_pull_request_revision(repo="", number=42)


def test_reader_returns_stable_pull_request_details() -> None:
    pull = SimpleNamespace(
        number=42,
        title="Keep the flock together",
        state="open",
        user=FakeUser(login="alice"),
        created_at=None,
        updated_at=None,
        html_url="https://github.com/acme/widgets/pull/42",
        body="Protect the public flock invariant.",
        base=SimpleNamespace(ref="main"),
        head=SimpleNamespace(ref="keep-flock-together"),
        draft=False,
        merged=False,
        mergeable_state="clean",
        changed_files=2,
        additions=14,
        deletions=3,
        labels=[SimpleNamespace(name="architecture")],
    )
    client = FakeGithubClient(FakeRepository([], pull_details=pull))
    reader = GitHubPullRequestReader(client=client, default_repo="acme/widgets")

    result = reader.get_pull_request(number=42, repo="")

    assert result == {
        "number": 42,
        "title": "Keep the flock together",
        "state": "open",
        "author": "alice",
        "created_at": None,
        "updated_at": None,
        "url": "https://github.com/acme/widgets/pull/42",
        "repo": "acme/widgets",
        "body": "Protect the public flock invariant.",
        "base": "main",
        "head": "keep-flock-together",
        "draft": False,
        "merged": False,
        "mergeable_state": "clean",
        "changed_files": 2,
        "additions": 14,
        "deletions": 3,
        "labels": ["architecture"],
    }


def test_reader_returns_each_reviewers_effective_review_state() -> None:
    reviews = [
        SimpleNamespace(
            user=FakeUser(login="alice"),
            state="CHANGES_REQUESTED",
            submitted_at=None,
            body="Please protect the invariant.",
        ),
        SimpleNamespace(
            user=FakeUser(login="alice"),
            state="APPROVED",
            submitted_at=None,
            body="Fixed now.",
        ),
        SimpleNamespace(
            user=FakeUser(login="bob"),
            state="COMMENTED",
            submitted_at=None,
            body="One non-blocking thought.",
        ),
    ]
    pull = SimpleNamespace(
        number=42,
        state="open",
        merged=False,
        mergeable_state="clean",
        get_reviews=lambda: reviews,
        get_review_requests=lambda: ([FakeUser(login="carol")], []),
    )
    client = FakeGithubClient(FakeRepository([], pull_details=pull))
    reader = GitHubPullRequestReader(client=client, default_repo="acme/widgets")

    result = reader.get_pull_request_reviews(number=42, repo="")

    assert result == {
        "repo": "acme/widgets",
        "number": 42,
        "state": "open",
        "merged": False,
        "mergeable_state": "clean",
        "approved_by": ["alice"],
        "changes_requested_by": [],
        "review_requested_from": ["carol"],
        "review_count": 3,
        "reviews": [
            {
                "author": "alice",
                "state": "CHANGES_REQUESTED",
                "submitted_at": None,
                "body": "Please protect the invariant.",
            },
            {
                "author": "alice",
                "state": "APPROVED",
                "submitted_at": None,
                "body": "Fixed now.",
            },
            {
                "author": "bob",
                "state": "COMMENTED",
                "submitted_at": None,
                "body": "One non-blocking thought.",
            },
        ],
    }


def test_reader_returns_pull_request_base_and_head_revisions() -> None:
    pull = SimpleNamespace(
        number=42,
        title="Keep the flock together",
        body="Fixes the gate latch.",
        base=SimpleNamespace(sha="base123"),
        head=SimpleNamespace(sha="abc123"),
    )
    repository = FakeRepository([], pull_details=pull)
    client = FakeGithubClient(repository)
    reader = GitHubPullRequestReader(client=client, default_repo="acme/widgets")

    revision = reader.get_pull_request_revision(repo="", number=42)

    assert repository.requested_pulls == [42]
    assert revision == PullRequestRevision(
        repo="acme/widgets",
        number=42,
        base_sha="base123",
        head_sha="abc123",
        title="Keep the flock together",
        body="Fixes the gate latch.",
    )


def test_reader_truncates_long_review_bodies_and_defaults_missing_text() -> None:
    long_pull = SimpleNamespace(
        number=42,
        title="Long",
        body="x" * 5000,
        base=SimpleNamespace(sha="base123"),
        head=SimpleNamespace(sha="abc123"),
    )
    empty_pull = SimpleNamespace(
        number=42,
        title=None,
        body=None,
        base=SimpleNamespace(sha="base123"),
        head=SimpleNamespace(sha="abc123"),
    )

    long_revision = GitHubPullRequestReader(
        client=FakeGithubClient(FakeRepository([], pull_details=long_pull)),
        default_repo="acme/widgets",
    ).get_pull_request_revision(repo="", number=42)
    empty_revision = GitHubPullRequestReader(
        client=FakeGithubClient(FakeRepository([], pull_details=empty_pull)),
        default_repo="acme/widgets",
    ).get_pull_request_revision(repo="", number=42)

    assert long_revision.body == "x" * 4000
    assert (empty_revision.title, empty_revision.body) == ("", "")


def test_reader_fetches_one_stable_manifest_snapshot() -> None:
    changed_file = SimpleNamespace(
        filename="src/example.py",
        status="modified",
        additions=2,
        deletions=1,
        patch="@@ -1 +1,2 @@\n-old\n+new\n+line",
        previous_filename=None,
    )
    pull = SimpleNamespace(
        number=42,
        title="Keep the flock together",
        body=None,
        base=SimpleNamespace(sha="base123"),
        head=SimpleNamespace(sha="head456"),
        get_files=lambda: [changed_file],
    )
    repository = FakeRepository([], pull_details=pull)
    reader = GitHubPullRequestReader(
        client=FakeGithubClient(repository),
        default_repo="acme/widgets",
    )

    snapshot = reader.fetch_snapshot(repo="", number=42)

    assert repository.requested_pulls == [42, 42]
    assert snapshot == PullRequestSnapshot(
        repo="acme/widgets",
        number=42,
        base_sha="base123",
        head_sha="head456",
        title="Keep the flock together",
        files=[
            SnapshotFile(
                path="src/example.py",
                status="modified",
                additions=2,
                deletions=1,
                patch="@@ -1 +1,2 @@\n-old\n+new\n+line",
            )
        ],
    )


def test_reader_rejects_a_snapshot_when_the_pull_request_head_changes() -> None:
    first_pull = SimpleNamespace(
        number=42,
        base=SimpleNamespace(sha="base123"),
        head=SimpleNamespace(sha="head456"),
        get_files=list,
    )
    second_pull = SimpleNamespace(
        number=42,
        base=SimpleNamespace(sha="base123"),
        head=SimpleNamespace(sha="new-head"),
    )

    class ChangingRepository(FakeRepository):
        def __init__(self) -> None:
            super().__init__([], pull_details=first_pull)
            self._calls = 0

        def get_pull(self, number: int) -> Any:
            self._calls += 1
            assert number == 42
            return first_pull if self._calls == 1 else second_pull

    reader = GitHubPullRequestReader(
        client=FakeGithubClient(ChangingRepository()),
        default_repo="acme/widgets",
    )

    with pytest.raises(RuntimeError, match="head changed"):
        reader.fetch_snapshot(repo="", number=42)


def test_reader_propagates_snapshot_api_failures() -> None:
    class FailingPull:
        number = 42
        base = SimpleNamespace(sha="base123")
        head = SimpleNamespace(sha="head456")

        def get_files(self) -> list[Any]:
            raise RuntimeError("GitHub file listing unavailable")

    repository = FakeRepository([], pull_details=FailingPull())
    reader = GitHubPullRequestReader(
        client=FakeGithubClient(repository),
        default_repo="acme/widgets",
    )

    with pytest.raises(RuntimeError, match="file listing unavailable"):
        reader.fetch_snapshot(repo="", number=42)


def test_reader_normalizes_missing_patches_to_empty_strings() -> None:
    changed_file = SimpleNamespace(
        filename="renamed.py",
        status="renamed",
        additions=0,
        deletions=0,
        patch=None,
        previous_filename="example.py",
    )
    pull = SimpleNamespace(
        number=42,
        title="Keep the flock together",
        body=None,
        base=SimpleNamespace(sha="base123"),
        head=SimpleNamespace(sha="head456"),
        get_files=lambda: [changed_file],
    )
    repository = FakeRepository([], pull_details=pull)
    reader = GitHubPullRequestReader(
        client=FakeGithubClient(repository),
        default_repo="acme/widgets",
    )

    snapshot = reader.fetch_snapshot(repo="", number=42)

    assert snapshot.files[0].patch == ""
    assert snapshot.files[0].previous_path == "example.py"
