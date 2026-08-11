"""A conversational LangGraph that routes Inquiry and Review intents."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal, NotRequired, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from review_sheep.domain import (
    ChatIntent,
    InquiryAnswer,
    IntentDecision,
    Review,
    ReviewError,
)
from review_sheep.inquiry import InquiryAgent
from review_sheep.intent import IntentClassifier
from review_sheep.report import render_report

logger = logging.getLogger(__name__)


class Reviewer(Protocol):
    """Run a Review regardless of how its changed code was prepared."""

    def review(self, *, repo: str, number: int) -> Review | ReviewError: ...


class ChatState(MessagesState):
    """Conversation messages plus the latest route and structured result."""

    intent: NotRequired[IntentDecision]
    answer: NotRequired[InquiryAnswer]
    review: NotRequired[Review | ReviewError]


class InquiryChatState(ChatState):
    """Backward-compatible name for the original Inquiry-only chat state."""


class _InquiryOnlyClassifier:
    def classify(self, messages: Sequence[BaseMessage]) -> IntentDecision:
        return IntentDecision(intent=ChatIntent.INQUIRY)


def create_chatbot_graph(
    *,
    agent: InquiryAgent,
    classifier: IntentClassifier | None = None,
    reviewer: Reviewer | None = None,
) -> CompiledStateGraph[
    ChatState,
    None,
    ChatState,
    ChatState,
]:
    """Build an intent-routed LangGraph over Inquiry and Review agents."""
    intent_classifier = classifier or _InquiryOnlyClassifier()

    def classify_intent(state: ChatState) -> dict[str, object]:
        if not state["messages"]:
            return {}
        decision = intent_classifier.classify(state["messages"])
        previous = state.get("intent")
        if (
            decision.intent is ChatIntent.REVIEW
            and previous is not None
            and previous.intent is ChatIntent.REVIEW
        ):
            decision = decision.model_copy(
                update={
                    "repo": decision.repo or previous.repo,
                    "pull_request_number": (
                        decision.pull_request_number or previous.pull_request_number
                    ),
                }
            )
        logger.info(
            "chat_graph.route intent=%s repo=%s pr=%s",
            decision.intent.value,
            decision.repo or "none",
            decision.pull_request_number or "none",
        )
        return {"intent": decision}

    def route(state: ChatState) -> Literal["inquiry", "review", "unrelated"]:
        decision = state.get("intent")
        if decision is not None and decision.intent is ChatIntent.REVIEW:
            return "review"
        if decision is not None and decision.intent is ChatIntent.UNRELATED:
            return "unrelated"
        return "inquiry"

    def bot(state: ChatState) -> dict[str, object]:
        if not state["messages"]:
            return _reply("What would you like to know or review about pull requests?")

        decision = state.get("intent")
        if decision is not None and decision.intent is ChatIntent.UNKNOWN:
            return _reply(
                f"Intent classification failed: {decision.error or 'unknown error'}"
            )

        logger.info("chat_graph.inquiry.start")
        answer = agent.invoke(state["messages"])
        response = answer.text or f"Inquiry failed: {answer.error}"
        return {
            "messages": [AIMessage(content=response)],
            "answer": answer,
        }

    def review_bot(state: ChatState) -> dict[str, object]:
        decision = state["intent"]
        if reviewer is None:
            return _reply("Changed-code Review is not configured for this chatbot.")
        if decision.repo is None:
            return _reply("Which repository should I review? Enter it as owner/name.")
        if decision.pull_request_number is None:
            return _reply(
                f"Which pull-request number should I review in {decision.repo}?"
            )

        logger.info(
            "chat_graph.review.start repo=%s pr=%d",
            decision.repo,
            decision.pull_request_number,
        )
        result = reviewer.review(
            repo=decision.repo,
            number=decision.pull_request_number,
        )
        if isinstance(result, Review):
            response = render_report(result).text
            logger.info("chat_graph.review.complete findings=%d", len(result.findings))
        else:
            logger.warning(
                "chat_graph.review.failed operation=%s type=%s message=%s",
                result.operation.value,
                result.error_type,
                result.message,
            )
            response = (
                f"Review failed during {result.operation.value}: "
                f"{result.error_type}: {result.message}"
            )
        return {
            "messages": [AIMessage(content=response)],
            "review": result,
        }

    def unrelated_bot(state: ChatState) -> dict[str, object]:
        return _reply(
            "I can only help with pull-request metadata or changed-code reviews."
        )

    graph = StateGraph(ChatState)
    graph.add_node("IntentClassifier", classify_intent)
    graph.add_node("Bot", bot)
    graph.add_node("ReviewBot", review_bot)
    graph.add_node("UnrelatedBot", unrelated_bot)
    graph.add_edge(START, "IntentClassifier")
    graph.add_conditional_edges(
        "IntentClassifier",
        route,
        {
            "inquiry": "Bot",
            "review": "ReviewBot",
            "unrelated": "UnrelatedBot",
        },
    )
    graph.add_edge("Bot", END)
    graph.add_edge("ReviewBot", END)
    graph.add_edge("UnrelatedBot", END)
    return graph.compile()


def _reply(message: str, **state: object) -> dict[str, object]:
    return {"messages": [AIMessage(content=message)], **state}
