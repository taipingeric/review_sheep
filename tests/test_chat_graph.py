from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from review_sheep import (
    ChatIntent,
    Finding,
    InquiryAnswer,
    InquiryChatState,
    IntentDecision,
    ManifestReviewAgent,
    PullRequestSnapshot,
    Review,
    ReviewCheckout,
    ReviewWorkspace,
    create_chatbot_graph,
    create_inquiry_agent,
    create_review_agent,
)


class ScriptedInquiryModel(BaseChatModel):
    responses: list[AIMessage]
    response_index: int = 0
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-inquiry-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedInquiryModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append(messages)
        response = self.responses[self.response_index]
        self.response_index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class FailingInquiryModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing-inquiry-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FailingInquiryModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("gateway unavailable")


class FakePullRequestReader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_pull_requests(
        self, *, state: str, limit: int, repo: str
    ) -> dict[str, Any]:
        self.calls.append(
            ("list_pull_requests", {"state": state, "limit": limit, "repo": repo})
        )
        return {
            "repo": repo,
            "state": state,
            "count": 1,
            "truncated": False,
            "pull_requests": [
                {
                    "number": 42,
                    "title": "Keep the flock together",
                    "url": "https://github.com/acme/widgets/pull/42",
                }
            ],
        }

    def get_pull_request(self, *, number: int, repo: str) -> dict[str, Any]:
        raise AssertionError("not expected in this chat")

    def get_pull_request_reviews(self, *, number: int, repo: str) -> dict[str, Any]:
        raise AssertionError("not expected in this chat")


class ScriptedIntentClassifier:
    def __init__(self, *decisions: IntentDecision) -> None:
        self.decisions = iter(decisions)

    def classify(self, messages: Sequence[BaseMessage]) -> IntentDecision:
        return next(self.decisions)


class EmptyCheckoutSource:
    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout:
        return ReviewCheckout(
            repo=repo,
            pull_request_number=number,
            base_sha="base123",
            head_sha="abc123",
            root=Path.cwd(),
        )


class EmptyReviewRunner:
    def run(self, checkout: ReviewCheckout) -> list[Finding]:
        return []


class EmptySnapshotSource:
    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        return PullRequestSnapshot(
            repo=repo,
            number=number,
            base_sha="base123",
            head_sha="abc123",
            files=[],
        )


class EmptyManifestRunner:
    def run(self, workspace: ReviewWorkspace) -> list[Finding]:
        return []


class RecordingChainHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.names: list[str] = []

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        if name:
            self.names.append(name)


def test_chatbot_graph_routes_conversation_through_the_inquiry_agent() -> None:
    model = ScriptedInquiryModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_pull_requests",
                        "args": {
                            "state": "open",
                            "limit": 10,
                            "repo": "acme/widgets",
                        },
                        "id": "call-list",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "Open pull request: #42 Keep the flock together — "
                    "https://github.com/acme/widgets/pull/42"
                )
            ),
            AIMessage(
                content=("The newest is #42 — https://github.com/acme/widgets/pull/42")
            ),
        ]
    )
    github = FakePullRequestReader()
    agent = create_inquiry_agent(model=model, github=github)
    chatbot = create_chatbot_graph(agent=agent)

    state = cast(
        InquiryChatState,
        chatbot.invoke(InquiryChatState(messages=[])),
    )
    assert state["messages"][-1].content == (
        "What would you like to know or review about pull requests?"
    )

    state["messages"] = [
        *state["messages"],
        HumanMessage(content="Which pull requests are open in acme/widgets?"),
    ]
    state = cast(InquiryChatState, chatbot.invoke(state))

    assert state["answer"] == InquiryAnswer(
        text=(
            "Open pull request: #42 Keep the flock together — "
            "https://github.com/acme/widgets/pull/42"
        )
    )
    assert github.calls == [
        (
            "list_pull_requests",
            {"state": "open", "limit": 10, "repo": "acme/widgets"},
        )
    ]

    state["messages"] = [
        *state["messages"],
        HumanMessage(content="Which one is newest?"),
    ]
    state = cast(InquiryChatState, chatbot.invoke(state))

    assert state["answer"].text == (
        "The newest is #42 — https://github.com/acme/widgets/pull/42"
    )
    assert any(
        message.text == "Which pull requests are open in acme/widgets?"
        for message in model.seen_messages[-1]
    )


def test_chatbot_graph_returns_inquiry_failures_as_structured_chat_state() -> None:
    agent = create_inquiry_agent(
        model=FailingInquiryModel(),
        github=FakePullRequestReader(),
    )
    chatbot = create_chatbot_graph(agent=agent)

    result = chatbot.invoke(
        {"messages": [HumanMessage(content="Which pull requests are open?")]}
    )

    assert result["answer"] == InquiryAnswer(error="RuntimeError: gateway unavailable")
    assert result["messages"][-1].content == (
        "Inquiry failed: RuntimeError: gateway unavailable"
    )


def test_chatbot_graph_routes_changed_code_intent_to_review_bot() -> None:
    inquiry_agent = create_inquiry_agent(
        model=ScriptedInquiryModel(responses=[]),
        github=FakePullRequestReader(),
    )
    reviewer = create_review_agent(
        source=EmptyCheckoutSource(),
        runner=EmptyReviewRunner(),
    )
    chatbot = create_chatbot_graph(
        agent=inquiry_agent,
        classifier=ScriptedIntentClassifier(
            IntentDecision(
                intent=ChatIntent.REVIEW,
                repo="acme/widgets",
                pull_request_number=42,
            )
        ),
        reviewer=reviewer,
    )

    result = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content="Review the changed code in acme/widgets#42.")
            ]
        }
    )

    assert result["intent"].intent is ChatIntent.REVIEW
    assert isinstance(result["review"], Review)
    assert (
        result["messages"][-1].content
        == """# Review Report: acme/widgets#42

Head: `abc123`
Findings: 0

No Findings.
"""
    )


def test_chat_trace_callback_propagates_into_manifest_review_graph() -> None:
    handler = RecordingChainHandler()
    reviewer = ManifestReviewAgent(
        source=EmptySnapshotSource(),
        runner=EmptyManifestRunner(),
    )
    chatbot = create_chatbot_graph(
        agent=create_inquiry_agent(
            model=ScriptedInquiryModel(responses=[]),
            github=FakePullRequestReader(),
        ),
        classifier=ScriptedIntentClassifier(
            IntentDecision(
                intent=ChatIntent.REVIEW,
                repo="langchain-ai/langgraph",
                pull_request_number=8569,
            )
        ),
        reviewer=reviewer,
    )

    result = chatbot.invoke(
        {"messages": [HumanMessage(content="Review langchain-ai/langgraph#8569")]},
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_session_id": "session-123"},
        },
    )

    assert isinstance(result["review"], Review)
    assert {"fetch_snapshot", "prepare_workspace", "run_review"}.issubset(handler.names)


def test_chatbot_graph_does_not_answer_an_unrelated_message() -> None:
    chatbot = create_chatbot_graph(
        agent=create_inquiry_agent(
            model=ScriptedInquiryModel(responses=[]),
            github=FakePullRequestReader(),
        ),
        classifier=ScriptedIntentClassifier(
            IntentDecision(intent=ChatIntent.UNRELATED)
        ),
    )

    result = chatbot.invoke(
        {"messages": [HumanMessage(content="What is the weather?")]}
    )

    assert result["intent"].intent is ChatIntent.UNRELATED
    assert result["messages"][-1].content == (
        "I can only help with pull-request metadata or changed-code reviews."
    )
    assert "answer" not in result
    assert "review" not in result


def test_chatbot_graph_routes_identity_questions_to_inquiry_agent() -> None:
    chatbot = create_chatbot_graph(
        agent=create_inquiry_agent(
            model=ScriptedInquiryModel(
                responses=[
                    AIMessage(
                        content=(
                            "I am Review Sheep, an assistant for understanding and "
                            "reviewing GitHub pull requests."
                        )
                    )
                ]
            ),
            github=FakePullRequestReader(),
        ),
        classifier=ScriptedIntentClassifier(IntentDecision(intent=ChatIntent.INQUIRY)),
    )

    result = chatbot.invoke({"messages": [HumanMessage(content="Who are you?")]})

    assert result["intent"].intent is ChatIntent.INQUIRY
    assert result["messages"][-1].content == (
        "I am Review Sheep, an assistant for understanding and reviewing "
        "GitHub pull requests."
    )


def test_review_route_collects_missing_context_across_turns() -> None:
    reviewer = create_review_agent(
        source=EmptyCheckoutSource(),
        runner=EmptyReviewRunner(),
    )
    chatbot = create_chatbot_graph(
        agent=create_inquiry_agent(
            model=ScriptedInquiryModel(responses=[]),
            github=FakePullRequestReader(),
        ),
        classifier=ScriptedIntentClassifier(
            IntentDecision(intent=ChatIntent.REVIEW, repo="acme/widgets"),
            IntentDecision(intent=ChatIntent.REVIEW, pull_request_number=42),
        ),
        reviewer=reviewer,
    )

    state = cast(
        InquiryChatState,
        chatbot.invoke(
            {"messages": [HumanMessage(content="Review changed code in acme/widgets.")]}
        ),
    )
    assert state["messages"][-1].content == (
        "Which pull-request number should I review in acme/widgets?"
    )

    state["messages"] = [
        *state["messages"],
        HumanMessage(content="42"),
    ]
    state = cast(InquiryChatState, chatbot.invoke(state))

    assert state["intent"] == IntentDecision(
        intent=ChatIntent.REVIEW,
        repo="acme/widgets",
        pull_request_number=42,
    )
    assert isinstance(state["review"], Review)


def test_chatbot_graph_routes_through_intent_classifier_node() -> None:
    agent = create_inquiry_agent(
        model=ScriptedInquiryModel(responses=[]),
        github=FakePullRequestReader(),
    )
    chatbot = create_chatbot_graph(agent=agent)

    assert {(edge.source, edge.target) for edge in chatbot.get_graph().edges} == {
        ("__start__", "IntentClassifier"),
        ("IntentClassifier", "Bot"),
        ("IntentClassifier", "ReviewBot"),
        ("IntentClassifier", "UnrelatedBot"),
        ("Bot", "__end__"),
        ("ReviewBot", "__end__"),
        ("UnrelatedBot", "__end__"),
    }
