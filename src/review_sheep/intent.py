"""LangChain agent for routing chat turns to Inquiry or Review."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field

from review_sheep.domain import ChatIntent, IntentDecision

SYSTEM_PROMPT = (
    "Classify the user's latest pull-request request into exactly one route.\n"
    "- inquiry: metadata such as lists, titles, state, authors, approvals, review "
    "status, or URLs; also greetings and questions about Review Sheep's identity or "
    "capabilities.\n"
    "- review: inspect changed code or diffs for correctness, security, conventions, "
    "tests, or actionable Findings.\n"
    "- unrelated: requests unrelated to pull requests and changed-code review, except "
    "for greetings and questions about Review Sheep itself.\n"
    "Use the full conversation to resolve follow-up answers. For review, extract an "
    "explicit repository in owner/name form and a positive pull-request number when "
    "they are present. Never invent either value."
)


class _RoutableIntentDecision(BaseModel):
    """The routes a model may select; UNKNOWN is reserved for runtime failure."""

    model_config = ConfigDict(frozen=True)

    intent: Literal["inquiry", "review", "unrelated"]
    repo: str | None = Field(default=None, pattern=r"^[^/\s]+/[^/\s]+$")
    pull_request_number: int | None = Field(default=None, gt=0)


class IntentClassifier(Protocol):
    """Classify accumulated messages into one structured agent route."""

    def classify(self, messages: Sequence[BaseMessage]) -> IntentDecision: ...


class _LangChainIntentClassifier:
    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def classify(self, messages: Sequence[BaseMessage]) -> IntentDecision:
        """Return stable routing data even when classification fails."""
        if not messages:
            return IntentDecision(intent=ChatIntent.UNKNOWN, error="No messages")
        try:
            state = self._agent.invoke({"messages": list(messages)})
            response = state["structured_response"]
            return IntentDecision.model_validate(response.model_dump())
        except Exception as error:  # noqa: BLE001 - routing failure is public data
            return IntentDecision(
                intent=ChatIntent.UNKNOWN,
                error=f"{type(error).__name__}: {error}",
            )


def create_intent_classifier(*, model: str | BaseChatModel) -> IntentClassifier:
    """Create the structured LangChain intent-classification agent."""
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(_RoutableIntentDecision),
        system_prompt=SYSTEM_PROMPT,
    )
    return _LangChainIntentClassifier(agent)
