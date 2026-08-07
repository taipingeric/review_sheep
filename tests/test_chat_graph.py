from __future__ import annotations

from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from review_sheep import (
    PullRequestSnapshot,
    Review,
    ReviewChatState,
    ReviewError,
    create_chatbot_graph,
)


class DeterministicSnapshotSource:
    def get_pull_request(self, *, repo: str, number: int) -> dict[str, Any]:
        return {
            "repo": repo,
            "number": number,
            "state": "open",
            "title": "Keep the flock together",
            "url": f"https://github.com/{repo}/pull/{number}",
        }

    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        return PullRequestSnapshot(
            repo=repo,
            number=number,
            head_sha="abc123",
            files=[],
        )


class FailingSnapshotSource:
    def get_pull_request(self, *, repo: str, number: int) -> dict[str, Any]:
        return {
            "repo": repo,
            "number": number,
            "state": "open",
            "title": "Keep the flock together",
            "url": f"https://github.com/{repo}/pull/{number}",
        }

    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        raise RuntimeError("GitHub is unavailable")


class DeterministicReviewModel(BaseChatModel):
    structured_tool_name: str | None = None
    seen_prompts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "deterministic-review-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> DeterministicReviewModel:
        names = [
            tool.name if hasattr(tool, "name") else tool["function"]["name"]
            for tool in tools
        ]
        structured_tool_name = next(
            name
            for name in names
            if name
            in {
                "CorrectnessResult",
                "SecurityResult",
                "ConventionsAndTestsResult",
            }
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
        prompt = next(
            str(message.text)
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        self.seen_prompts.append(prompt)
        assert self.structured_tool_name is not None
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


def test_chatbot_graph_runs_the_review_agent_from_start_to_bot_to_end() -> None:
    model = DeterministicReviewModel()
    chatbot = create_chatbot_graph(
        source=DeterministicSnapshotSource(),
        model=model,
    )

    result = cast(
        ReviewChatState,
        chatbot.invoke(ReviewChatState(messages=[])),
    )
    assert result["messages"][-1].content == (
        "Which repository should I review? Enter it as owner/name."
    )

    result["messages"] = [
        *result["messages"],
        HumanMessage(content="acme/widgets"),
    ]
    result = cast(
        ReviewChatState,
        chatbot.invoke(result),
    )
    assert result["repo"] == "acme/widgets"
    assert result["messages"][-1].content == (
        "Which open pull-request number should I review in acme/widgets?"
    )

    result["messages"] = [
        *result["messages"],
        HumanMessage(content="42"),
    ]
    result = cast(
        ReviewChatState,
        chatbot.invoke(result),
    )
    assert result["pull_request_number"] == 42
    assert result["messages"][-1].content == (
        "What should the Review focus on?"
    )

    result["messages"] = [
        *result["messages"],
        HumanMessage(content="Focus on authorization regressions."),
    ]
    result = cast(
        ReviewChatState,
        chatbot.invoke(result),
    )

    assert isinstance(result["review"], Review)
    assert result["review"].repo == "acme/widgets"
    assert result["review"].pull_request_number == 42
    assert result["review"].findings == []
    assert result["messages"][-1].content == """# Review Report: acme/widgets#42

Head: `abc123`
Findings: 0

No Findings.
"""
    assert len(model.seen_prompts) == 3
    assert all(
        "Caller instructions:\nFocus on authorization regressions." in prompt
        for prompt in model.seen_prompts
    )
    assert {
        (edge.source, edge.target) for edge in chatbot.get_graph().edges
    } == {
        ("__start__", "Bot"),
        ("Bot", "__end__"),
    }


def test_chatbot_graph_returns_review_failures_as_structured_chat_state() -> None:
    chatbot = create_chatbot_graph(
        source=FailingSnapshotSource(),
        model=DeterministicReviewModel(),
    )

    result = chatbot.invoke(
        {
            "repo": "acme/widgets",
            "pull_request_number": 42,
            "messages": [HumanMessage(content="Review correctness.")],
        }
    )

    assert isinstance(result["review"], ReviewError)
    assert result["messages"][-1].content == (
        "Review failed during fetch_snapshot: RuntimeError: "
        "GitHub is unavailable"
    )
