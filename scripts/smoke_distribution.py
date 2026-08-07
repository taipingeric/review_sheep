"""Exercise the installed public package with deterministic collaborators."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from review_sheep import (
    Confidence,
    Finding,
    InquiryAnswer,
    Lens,
    Location,
    PullRequestSnapshot,
    Review,
    ReviewChatState,
    ReviewWorkspace,
    Severity,
    SnapshotFile,
    create_chatbot_graph,
    create_inquiry,
    create_review_agent,
    render_report,
)


class DeterministicModel(BaseChatModel):
    """Return one fixed Inquiry answer without a provider or credentials."""

    structured_tool_name: str | None = None

    @property
    def _llm_type(self) -> str:
        return "distribution-smoke-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> DeterministicModel:
        names = [
            tool.name if hasattr(tool, "name") else tool["function"]["name"]
            for tool in tools
        ]
        structured_tool_name = next(
            (
                name
                for name in names
                if name
                in {
                    "CorrectnessResult",
                    "SecurityResult",
                    "ConventionsAndTestsResult",
                }
            ),
            None,
        )
        return self.model_copy(
            update={"structured_tool_name": structured_tool_name}
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.structured_tool_name is not None:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": self.structured_tool_name,
                                    "args": {"findings": []},
                                    "id": f"call-{self.structured_tool_name}",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
            )
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
    def get_pull_request(self, *, repo: str, number: int) -> dict[str, Any]:
        return {
            "number": number,
            "state": "open",
            "title": "Deterministic review",
            "html_url": f"https://github.com/{repo}/pull/{number}",
        }

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
    chatbot = create_chatbot_graph(
        source=DeterministicSnapshotSource(),
        model=DeterministicModel(),
    )
    chat_state = cast(
        ReviewChatState,
        chatbot.invoke(ReviewChatState(messages=[])),
    )
    assert "repository" in str(chat_state["messages"][-1].content)
    for reply in ("acme/widgets", "42", "Review correctness."):
        chat_state["messages"] = [
            *chat_state["messages"],
            HumanMessage(content=reply),
        ]
        chat_state = cast(ReviewChatState, chatbot.invoke(chat_state))
    assert isinstance(chat_state["review"], Review)
    assert str(chat_state["messages"][-1].content).endswith("No Findings.\n")

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
