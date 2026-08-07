"""The lightweight, read-only Inquiry path."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from review_sheep.domain import InquiryAnswer

SYSTEM_PROMPT = (
    "You are Review Sheep, a read-only assistant for understanding and reviewing "
    "GitHub pull requests.\n"
    "- For greetings or questions about your identity and capabilities, briefly "
    "introduce yourself as Review Sheep and explain that you can answer pull-request "
    "metadata questions or perform changed-code Reviews. Do not call a tool unless "
    "the user asks for pull-request information.\n"
    "- Answer pull-request questions using only metadata returned by the available "
    "tools.\n"
    "- Cite pull request numbers and URLs.\n"
    "- If a tool returns truncated=true, explicitly say the list is incomplete.\n"
    "- Never guess at changed code, Findings, or unavailable metadata.\n"
    "- Reply in the language used by the user."
)


class PullRequestReader(Protocol):
    """Read-only GitHub metadata used by Inquiry."""

    def list_pull_requests(
        self, *, state: str, limit: int, repo: str
    ) -> dict[str, Any]: ...

    def get_pull_request(self, *, number: int, repo: str) -> dict[str, Any]: ...

    def get_pull_request_reviews(self, *, number: int, repo: str) -> dict[str, Any]: ...


class InquiryAgent:
    """Answer pull-request metadata questions through a LangChain agent."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def ask(self, question: str) -> InquiryAnswer:
        """Answer one standalone metadata-only question."""
        if not question.strip():
            return InquiryAnswer(error="Inquiry question must not be empty")

        return self.invoke([HumanMessage(content=question)])

    def invoke(self, messages: Sequence[BaseMessage]) -> InquiryAnswer:
        """Answer from accumulated conversation messages."""
        if not messages:
            return InquiryAnswer(error="Inquiry messages must not be empty")

        try:
            state = self._agent.invoke({"messages": list(messages)})
            result_messages = state["messages"]
            message = result_messages[-1]
        except Exception as error:  # noqa: BLE001 - errors are public data here
            return InquiryAnswer(error=f"{type(error).__name__}: {error}")

        if not isinstance(message, AIMessage) or not message.text:
            return InquiryAnswer(error="Inquiry produced no answer")
        return InquiryAnswer(
            text=message.text,
            incomplete=_contains_truncated_tool_data(result_messages),
        )


def _contains_truncated_tool_data(messages: list[BaseMessage]) -> bool:
    for message in messages:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("truncated") is True:
            return True
    return False


def _as_tool_data(
    operation: Any, *, operation_name: str, repo: str, **kwargs: Any
) -> str:
    try:
        return json.dumps(operation(repo=repo, **kwargs))
    except Exception as error:  # noqa: BLE001 - tool failures must not kill the run
        return json.dumps(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "operation": operation_name,
                "repo": repo,
            }
        )


def _tools(github: PullRequestReader) -> list[Any]:
    @tool
    def list_pull_requests(state: str = "open", limit: int = 10, repo: str = "") -> str:
        """List pull requests, most recently updated first.

        Args:
            state: Pull-request state: open, closed, or all.
            limit: Maximum number of pull requests to return.
            repo: Repository in owner/name form, or empty for the configured default.
        """
        if state not in {"open", "closed", "all"}:
            return json.dumps(
                {"error": f"invalid state {state!r}; use open, closed, or all"}
            )
        return _as_tool_data(
            github.list_pull_requests,
            operation_name="list_pull_requests",
            repo=repo,
            state=state,
            limit=max(1, min(limit, 50)),
        )

    @tool
    def get_pull_request(number: int, repo: str = "") -> str:
        """Get metadata for one pull request.

        Args:
            number: Pull-request number.
            repo: Repository in owner/name form, or empty for the configured default.
        """
        return _as_tool_data(
            github.get_pull_request,
            operation_name="get_pull_request",
            number=number,
            repo=repo,
        )

    @tool
    def get_pull_request_reviews(number: int, repo: str = "") -> str:
        """Get review-state metadata for one pull request.

        Args:
            number: Pull-request number.
            repo: Repository in owner/name form, or empty for the configured default.
        """
        return _as_tool_data(
            github.get_pull_request_reviews,
            operation_name="get_pull_request_reviews",
            number=number,
            repo=repo,
        )

    return [list_pull_requests, get_pull_request, get_pull_request_reviews]


def create_inquiry_agent(
    *, model: str | BaseChatModel, github: PullRequestReader
) -> InquiryAgent:
    """Create the read-only LangChain Inquiry agent from explicit adapters."""
    agent = create_agent(model=model, tools=_tools(github), system_prompt=SYSTEM_PROMPT)
    return InquiryAgent(agent)
