"""Read-only GitHub adapter shared by Inquiry and Review."""

from __future__ import annotations

from typing import Any

from review_sheep.domain import PullRequestRevision


class GitHubPullRequestReader:
    """Adapt PyGithub to metadata reads and pull-request revisions."""

    def __init__(self, *, client: Any, default_repo: str = "") -> None:
        self._client = client
        self._default_repo = default_repo

    def _repo(self, repo: str) -> str:
        target = repo or self._default_repo
        if not target:
            raise RuntimeError(
                "no repository given and no default repository configured"
            )
        return target

    @staticmethod
    def _summarize(pull: Any) -> dict[str, Any]:
        return {
            "number": pull.number,
            "title": pull.title,
            "state": pull.state,
            "author": pull.user.login if pull.user else None,
            "created_at": pull.created_at.isoformat() if pull.created_at else None,
            "updated_at": pull.updated_at.isoformat() if pull.updated_at else None,
            "url": pull.html_url,
        }

    def list_pull_requests(
        self, *, state: str, limit: int, repo: str
    ) -> dict[str, Any]:
        target = self._repo(repo)
        pulls = self._client.get_repo(target).get_pulls(
            state=state, sort="updated", direction="desc"
        )
        selected = list(pulls[: limit + 1])
        truncated = len(selected) > limit
        rows = [self._summarize(pull) for pull in selected[:limit]]
        return {
            "repo": target,
            "state": state,
            "count": len(rows),
            "truncated": truncated,
            "pull_requests": rows,
        }

    def get_pull_request(self, *, number: int, repo: str) -> dict[str, Any]:
        target = self._repo(repo)
        pull = self._client.get_repo(target).get_pull(number)
        details = self._summarize(pull)
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
        return details

    def get_pull_request_reviews(self, *, number: int, repo: str) -> dict[str, Any]:
        target = self._repo(repo)
        pull = self._client.get_repo(target).get_pull(number)

        reviews = []
        latest_by_user: dict[str, str] = {}
        for review in pull.get_reviews():
            author = review.user.login if review.user else None
            reviews.append(
                {
                    "author": author,
                    "state": review.state,
                    "submitted_at": (
                        review.submitted_at.isoformat() if review.submitted_at else None
                    ),
                    "body": (review.body or "")[:500],
                }
            )
            if author and review.state != "COMMENTED":
                latest_by_user[author] = review.state

        pending = [user.login for user in pull.get_review_requests()[0]]
        return {
            "repo": target,
            "number": number,
            "state": pull.state,
            "merged": pull.merged,
            "mergeable_state": pull.mergeable_state,
            "approved_by": [
                user for user, state in latest_by_user.items() if state == "APPROVED"
            ],
            "changes_requested_by": [
                user
                for user, state in latest_by_user.items()
                if state == "CHANGES_REQUESTED"
            ],
            "review_requested_from": pending,
            "review_count": len(reviews),
            "reviews": reviews,
        }

    def get_pull_request_revision(
        self, *, repo: str, number: int
    ) -> PullRequestRevision:
        """Read the immutable base and head commits expected by Review."""
        target = self._repo(repo)
        pull = self._client.get_repo(target).get_pull(number)
        return PullRequestRevision(
            repo=target,
            number=number,
            base_sha=pull.base.sha,
            head_sha=pull.head.sha,
        )
