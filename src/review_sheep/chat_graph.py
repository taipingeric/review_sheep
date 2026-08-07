"""A conversational LangGraph chatbot backed by the Review workflow."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, NotRequired, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from review_sheep.domain import Review, ReviewError
from review_sheep.report import render_report
from review_sheep.review import SnapshotSource, create_deep_review_agent

ReviewChatStage = Literal["repository", "pull_request", "instructions"]


class ReviewChatSource(SnapshotSource, Protocol):
    """Read pull-request metadata and one stable Review snapshot."""

    def get_pull_request(self, *, repo: str, number: int) -> dict[str, Any]: ...


class ReviewChatState(MessagesState):
    """Input messages and pull-request identity plus the completed Review."""

    repo: NotRequired[str]
    pull_request_number: NotRequired[int]
    awaiting: NotRequired[ReviewChatStage]
    review: NotRequired[Review | ReviewError]


def create_chatbot_graph(
    *,
    source: ReviewChatSource,
    model: str | BaseChatModel,
) -> CompiledStateGraph[
    ReviewChatState,
    None,
    ReviewChatState,
    ReviewChatState,
]:
    """Build a ``START -> Bot -> END`` graph around the deep Review agent."""

    def bot(state: ReviewChatState) -> dict[str, object]:
        if "repo" not in state and state.get("awaiting") != "repository":
            return _reply(
                "Which repository should I review? Enter it as owner/name.",
                awaiting="repository",
            )

        message = _latest_message_text(state["messages"])
        if "repo" not in state:
            repo = _repository_name(message)
            if repo is None:
                return _reply(
                    "That repository is invalid. Enter it as owner/name.",
                    awaiting="repository",
                )
            return _reply(
                f"Which open pull-request number should I review in {repo}?",
                repo=repo,
                awaiting="pull_request",
            )

        if "pull_request_number" not in state:
            if state.get("awaiting") != "pull_request":
                return _reply(
                    f"Which open pull-request number should I review in "
                    f"{state['repo']}?",
                    awaiting="pull_request",
                )
            number = _pull_request_number(message)
            if number is None:
                return _reply(
                    "That is not a positive pull-request number. Try again.",
                    awaiting="pull_request",
                )
            repo = state["repo"]
            try:
                details = source.get_pull_request(repo=repo, number=number)
            except Exception as error:  # noqa: BLE001 - external read boundary
                return _reply(
                    f"I could not read {repo}#{number}: "
                    f"{type(error).__name__}: {error}. Try another number.",
                    awaiting="pull_request",
                )
            if details["state"] != "open":
                return _reply(
                    f"{repo}#{number} is not open. Try another number.",
                    awaiting="pull_request",
                )
            return _reply(
                "What should the Review focus on?",
                pull_request_number=number,
                awaiting="instructions",
            )

        reviewer = create_deep_review_agent(
            source=source,
            model=model,
            instructions=message,
        )
        result = reviewer.review(
            repo=state["repo"],
            number=state["pull_request_number"],
        )
        if isinstance(result, Review):
            response = render_report(result).text
        else:
            response = (
                f"Review failed during {result.operation.value}: "
                f"{result.error_type}: {result.message}"
            )
        return {
            "messages": [AIMessage(content=response)],
            "review": result,
            "awaiting": "instructions",
        }

    graph = StateGraph(ReviewChatState)
    graph.add_node("Bot", bot)
    graph.add_edge(START, "Bot")
    graph.add_edge("Bot", END)
    return graph.compile()


def _latest_message_text(messages: Sequence[BaseMessage]) -> str:
    if not messages:
        raise ValueError("chatbot graph requires at least one message")
    text = str(messages[-1].text).strip()
    if not text:
        raise ValueError("latest chatbot message must contain text")
    return text


def _reply(message: str, **state: object) -> dict[str, object]:
    return {"messages": [AIMessage(content=message)], **state}


def _repository_name(message: str) -> str | None:
    parts = message.strip().split("/")
    if len(parts) != 2 or not all(part.strip() for part in parts):
        return None
    return "/".join(part.strip() for part in parts)


def _pull_request_number(message: str) -> int | None:
    try:
        number = int(message)
    except ValueError:
        return None
    return number if number > 0 else None
