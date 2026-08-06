from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, ValidationError

from review_sheep import InquiryAnswer, create_inquiry


class ScriptedToolCallingModel(BaseChatModel):
    responses: list[AIMessage]
    response_index: int = 0
    bound_tool_names: list[str] = Field(default_factory=list)
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolCallingModel:
        self.bound_tool_names = [tool.name for tool in tools]
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


class FakeGitHubReader:
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
        raise AssertionError("not expected in this Inquiry")

    def get_pull_request_reviews(self, *, number: int, repo: str) -> dict[str, Any]:
        raise AssertionError("not expected in this Inquiry")


class FailingGitHubReader(FakeGitHubReader):
    def list_pull_requests(
        self, *, state: str, limit: int, repo: str
    ) -> dict[str, Any]:
        self.calls.append(
            ("list_pull_requests", {"state": state, "limit": limit, "repo": repo})
        )
        raise RuntimeError("GitHub is unavailable")


class TruncatedGitHubReader(FakeGitHubReader):
    def list_pull_requests(
        self, *, state: str, limit: int, repo: str
    ) -> dict[str, Any]:
        result = super().list_pull_requests(state=state, limit=limit, repo=repo)
        result["truncated"] = True
        return result


def test_inquiry_answer_is_a_serializable_pydantic_contract() -> None:
    answer = InquiryAnswer(text="One open pull request.", incomplete=True)

    assert answer.model_dump() == {
        "text": "One open pull request.",
        "error": None,
        "incomplete": True,
    }


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"text": "One open pull request.", "error": "GitHub is unavailable"},
    ],
)
def test_inquiry_answer_requires_exactly_one_text_or_error(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        InquiryAnswer(**values)


def test_inquiry_answers_from_read_only_github_metadata() -> None:
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_pull_requests",
                        "args": {"state": "open", "limit": 10, "repo": "acme/widgets"},
                        "id": "call-1",
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
        ]
    )
    github = FakeGitHubReader()

    inquiry = create_inquiry(model=model, github=github)
    answer = inquiry.ask("Which pull requests are open in acme/widgets?")

    assert answer.text == (
        "Open pull request: #42 Keep the flock together — "
        "https://github.com/acme/widgets/pull/42"
    )
    assert answer.error is None
    assert github.calls == [
        (
            "list_pull_requests",
            {"state": "open", "limit": 10, "repo": "acme/widgets"},
        )
    ]
    assert model.bound_tool_names == [
        "list_pull_requests",
        "get_pull_request",
        "get_pull_request_reviews",
    ]


def test_inquiry_returns_invalid_tool_input_as_data_without_calling_github() -> None:
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_pull_requests",
                        "args": {
                            "state": "merged",
                            "limit": 10,
                            "repo": "acme/widgets",
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The supported states are open, closed, and all."),
        ]
    )
    github = FakeGitHubReader()

    answer = create_inquiry(model=model, github=github).ask(
        "Which pull requests are merged in acme/widgets?"
    )

    assert answer.text == "The supported states are open, closed, and all."
    assert answer.error is None
    assert github.calls == []
    tool_message = model.seen_messages[-1][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == (
        '{"error": "invalid state \'merged\'; use open, closed, or all"}'
    )


def test_inquiry_contains_github_failures_with_operation_and_repo_context() -> None:
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_pull_requests",
                        "args": {"state": "open", "limit": 10, "repo": "acme/widgets"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I could not read pull requests from acme/widgets."),
        ]
    )
    github = FailingGitHubReader()

    answer = create_inquiry(model=model, github=github).ask(
        "Which pull requests are open in acme/widgets?"
    )

    assert answer.text == "I could not read pull requests from acme/widgets."
    assert answer.error is None
    tool_message = model.seen_messages[-1][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == (
        '{"error": "GitHub is unavailable", "error_type": "RuntimeError", '
        '"operation": "list_pull_requests", "repo": "acme/widgets"}'
    )


def test_inquiry_marks_its_answer_incomplete_when_github_truncates_metadata() -> None:
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_pull_requests",
                        "args": {"state": "open", "limit": 1, "repo": "acme/widgets"},
                        "id": "call-1",
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
        ]
    )

    answer = create_inquiry(model=model, github=TruncatedGitHubReader()).ask(
        "Which pull request is open in acme/widgets?"
    )

    assert answer.incomplete is True
