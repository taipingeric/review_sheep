from __future__ import annotations

from io import StringIO
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from review_sheep.chat import main


class FakeGithubClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class DeterministicInquiryModel(BaseChatModel):
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)
    structured_tool_name: str | None = None

    @property
    def _llm_type(self) -> str:
        return "deterministic-inquiry-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> DeterministicInquiryModel:
        tool_names = [tool.name for tool in tools]
        structured_tool_name = next(
            (name for name in tool_names if "IntentDecision" in name),
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
        self.seen_messages.append(messages)
        if self.structured_tool_name is not None:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": self.structured_tool_name,
                                    "args": {"intent": "inquiry"},
                                    "id": "call-intent",
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
                            "Pull request #42 — https://github.com/acme/widgets/pull/42"
                        )
                    )
                )
            ]
        )


def _configure(monkeypatch: Any) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_chat_reports_github_client_startup_failure(monkeypatch: Any) -> None:
    _configure(monkeypatch)
    output = StringIO()
    errors = StringIO()

    def unavailable_github(_: str) -> Any:
        raise RuntimeError("GitHub credentials rejected")

    exit_code = main(
        input_fn=lambda _: "quit",
        output=output,
        error=errors,
        github_factory=unavailable_github,
        model_factory=lambda **_: DeterministicInquiryModel(),
    )

    assert exit_code == 1
    assert output.getvalue() == ""
    assert errors.getvalue() == "error: RuntimeError: GitHub credentials rejected\n"


def test_chat_exits_cleanly_on_eof_during_the_conversation(
    monkeypatch: Any,
) -> None:
    _configure(monkeypatch)
    output = StringIO()
    errors = StringIO()
    github = FakeGithubClient()

    def end_input(_: str) -> str:
        raise EOFError

    exit_code = main(
        input_fn=end_input,
        output=output,
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: DeterministicInquiryModel(),
    )

    assert exit_code == 0
    assert output.getvalue().startswith("Review Sheep chatbot ready;")
    assert (
        "What would you like to know or review about pull requests?"
        in output.getvalue()
    )
    assert errors.getvalue() == ""
    assert github.closed is True


def test_chat_treats_a_whitespace_base_url_as_unset(monkeypatch: Any) -> None:
    _configure(monkeypatch)
    monkeypatch.setenv("BASE_URL", "   ")
    base_urls: list[str | None] = []

    def capture_model_config(**kwargs: Any) -> DeterministicInquiryModel:
        base_urls.append(kwargs["base_url"])
        return DeterministicInquiryModel()

    github = FakeGithubClient()
    exit_code = main(
        input_fn=lambda _: "quit",
        output=StringIO(),
        error=StringIO(),
        github_factory=lambda _: github,
        model_factory=capture_model_config,
    )

    assert exit_code == 0
    assert base_urls == [None]
    assert github.closed is True


def test_chat_routes_each_question_through_inquiry_agent_and_langgraph(
    monkeypatch: Any,
) -> None:
    _configure(monkeypatch)
    github = FakeGithubClient()
    model = DeterministicInquiryModel()
    prompts = iter(["Tell me about pull request 42.", "quit"])
    output = StringIO()
    errors = StringIO()

    exit_code = main(
        input_fn=lambda _: next(prompts),
        output=output,
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: model,
    )

    assert exit_code == 0
    assert (
        "Bot: Pull request #42 — https://github.com/acme/widgets/pull/42\n"
        in output.getvalue()
    )
    assert any(
        isinstance(message, HumanMessage)
        and message.text == "Tell me about pull request 42."
        for message in model.seen_messages[-1]
    )
    assert errors.getvalue() == ""
    assert github.closed is True
