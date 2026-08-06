"""Exercise the installed public package with deterministic collaborators."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from review_sheep import (
    Confidence,
    Finding,
    InquiryAnswer,
    Lens,
    Location,
    PullRequestSnapshot,
    Review,
    ReviewWorkspace,
    Severity,
    SnapshotFile,
    create_inquiry,
    create_review_agent,
    render_report,
)


class DeterministicModel(BaseChatModel):
    """Return one fixed Inquiry answer without a provider or credentials."""

    @property
    def _llm_type(self) -> str:
        return "distribution-smoke-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> DeterministicModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=(
                            "Pull request #42: "
                            "https://github.com/acme/widgets/pull/42"
                        )
                    )
                )
            ]
        )


class UnusedGitHubReader:
    """Satisfy Inquiry construction while proving the model needs no live call."""

    def list_pull_requests(
        self, *, state: str, limit: int, repo: str
    ) -> dict[str, Any]:
        raise AssertionError("the smoke Inquiry must not call GitHub")

    def get_pull_request(self, *, number: int, repo: str) -> dict[str, Any]:
        raise AssertionError("the smoke Inquiry must not call GitHub")

    def get_pull_request_reviews(
        self, *, number: int, repo: str
    ) -> dict[str, Any]:
        raise AssertionError("the smoke Inquiry must not call GitHub")


class DeterministicSnapshotSource:
    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        return PullRequestSnapshot(
            repo=repo,
            number=number,
            head_sha="abc123",
            files=[
                SnapshotFile(
                    path="src/example.py",
                    status="modified",
                    additions=1,
                    deletions=1,
                    patch="@@ -1 +1 @@\n-old\n+new",
                )
            ],
        )


class DeterministicReviewRunner:
    def run(self, workspace: ReviewWorkspace) -> list[Finding]:
        assert workspace.read("/diffs/src/example.py.diff")
        return [
            Finding(
                description="The changed branch returns the wrong value.",
                location=Location(path="src/example.py", start_line=1, end_line=1),
                severity=Severity.HIGH,
                confidence=Confidence.CONFIRMED,
                lens=Lens.CORRECTNESS,
            )
        ]


def main() -> None:
    inquiry = create_inquiry(
        model=DeterministicModel(),
        github=UnusedGitHubReader(),
    )
    answer = inquiry.ask("Which pull request should I inspect?")
    assert answer == InquiryAnswer(
        text="Pull request #42: https://github.com/acme/widgets/pull/42"
    )

    result = create_review_agent(
        source=DeterministicSnapshotSource(),
        runner=DeterministicReviewRunner(),
    ).review(repo="acme/widgets", number=42)
    assert isinstance(result, Review)
    report = render_report(result)
    assert "src/example.py:1" in report.text
    print("distribution smoke passed")


if __name__ == "__main__":
    main()
