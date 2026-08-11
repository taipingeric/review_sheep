"""Exercise the installed public package with deterministic collaborators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from review_sheep import (
    Confidence,
    Finding,
    InquiryAnswer,
    InquiryChatState,
    Lens,
    Location,
    Review,
    ReviewCheckout,
    Severity,
    create_chatbot_graph,
    create_inquiry_agent,
    create_intent_classifier,
    create_manifest_review_agent,
    create_review_agent,
    render_report,
)
from review_sheep.chat import main as chat_main
from review_sheep.ci import main as ci_main


class DeterministicModel(BaseChatModel):
    """Return one fixed Inquiry answer without a provider or credentials."""

    structured_tool_name: str | None = None
    route_intent: str = "inquiry"

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
                if "IntentDecision" in name
                or name
                in {
                    "CorrectnessResult",
                    "SecurityResult",
                    "ConventionsAndTestsResult",
                }
            ),
            None,
        )
        return self.model_copy(update={"structured_tool_name": structured_tool_name})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.structured_tool_name is not None:
            if "IntentDecision" in self.structured_tool_name:
                args: dict[str, Any] = {"intent": self.route_intent}
                if self.route_intent == "review":
                    args.update(
                        {
                            "repo": "acme/widgets",
                            "pull_request_number": 42,
                        }
                    )
            else:
                args = {"findings": []}
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": self.structured_tool_name,
                                    "args": args,
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
                            "Pull request #42: https://github.com/acme/widgets/pull/42"
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

    def get_pull_request_reviews(self, *, number: int, repo: str) -> dict[str, Any]:
        raise AssertionError("the smoke Inquiry must not call GitHub")


class DeterministicCheckoutSource:
    def get_pull_request(self, *, repo: str, number: int) -> dict[str, Any]:
        return {
            "number": number,
            "state": "open",
            "title": "Deterministic review",
            "html_url": f"https://github.com/{repo}/pull/{number}",
        }

    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout:
        return ReviewCheckout(
            repo=repo,
            pull_request_number=number,
            base_sha="base123",
            head_sha="abc123",
            root=Path.cwd(),
        )


class DeterministicReviewRunner:
    def run(self, checkout: ReviewCheckout) -> list[Finding]:
        assert checkout.head_sha == "abc123"
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
    assert callable(chat_main)
    assert callable(ci_main)
    assert callable(create_manifest_review_agent)
    inquiry_agent = create_inquiry_agent(
        model=DeterministicModel(),
        github=UnusedGitHubReader(),
    )
    chatbot = create_chatbot_graph(
        agent=inquiry_agent,
        classifier=create_intent_classifier(
            model=DeterministicModel(route_intent="review")
        ),
        reviewer=create_review_agent(
            source=DeterministicCheckoutSource(),
            runner=DeterministicReviewRunner(),
        ),
    )
    chat_state = cast(
        InquiryChatState,
        chatbot.invoke(InquiryChatState(messages=[])),
    )
    assert "pull requests" in str(chat_state["messages"][-1].content)
    chat_state["messages"] = [
        *chat_state["messages"],
        HumanMessage(content="Review changed code in acme/widgets#42."),
    ]
    chat_state = cast(InquiryChatState, chatbot.invoke(chat_state))
    assert isinstance(chat_state["review"], Review)
    assert "src/example.py:1" in str(chat_state["messages"][-1].content)

    inquiry = create_inquiry_agent(
        model=DeterministicModel(),
        github=UnusedGitHubReader(),
    )
    answer = inquiry.ask("Which pull request should I inspect?")
    assert answer == InquiryAnswer(
        text="Pull request #42: https://github.com/acme/widgets/pull/42"
    )

    result = create_review_agent(
        source=DeterministicCheckoutSource(),
        runner=DeterministicReviewRunner(),
    ).review(repo="acme/widgets", number=42)
    assert isinstance(result, Review)
    report = render_report(result)
    assert "src/example.py:1" in report.text
    print("distribution smoke passed")


if __name__ == "__main__":
    main()
