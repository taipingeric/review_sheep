from __future__ import annotations

from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from review_sheep import (
    InquiryAnswer,
    InquiryChatState,
    create_chatbot_graph,
    create_inquiry_agent,
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
        "What would you like to know about pull requests?"
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


def test_chatbot_graph_has_one_langgraph_bot_node() -> None:
    agent = create_inquiry_agent(
        model=ScriptedInquiryModel(responses=[]),
        github=FakePullRequestReader(),
    )
    chatbot = create_chatbot_graph(agent=agent)

    assert {(edge.source, edge.target) for edge in chatbot.get_graph().edges} == {
        ("__start__", "Bot"),
        ("Bot", "__end__"),
    }
