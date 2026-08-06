from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from review_sheep import GitHubPullRequestReader


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

    def get_pulls(self, *, state: str, sort: str, direction: str) -> list[Any]:
        self.queries.append({"state": state, "sort": sort, "direction": direction})
        return self.pulls

    def get_pull(self, number: int) -> Any:
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
