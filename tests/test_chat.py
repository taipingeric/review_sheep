from __future__ import annotations

from io import StringIO
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from review_sheep.chat import main
from review_sheep.config import ChatConfig, LangfuseConfig


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


def _chat_config(*, base_url: str | None = None) -> ChatConfig:
    return ChatConfig(
        github_token="test-token",
        model="gpt-test",
        api_key="test-key",
        base_url=base_url,
    )


def test_chat_reports_github_client_startup_failure() -> None:
    output = StringIO()
    errors = StringIO()

    def unavailable_github(_: str) -> Any:
        raise RuntimeError("GitHub credentials rejected")

    exit_code = main(
        config=_chat_config(),
        input_fn=lambda _: "quit",
        output=output,
        error=errors,
        github_factory=unavailable_github,
        model_factory=lambda **_: DeterministicInquiryModel(),
    )

    assert exit_code == 1
    assert output.getvalue() == ""
    assert errors.getvalue() == "error: RuntimeError: GitHub credentials rejected\n"


def test_chat_exits_cleanly_on_eof_during_the_conversation() -> None:
    output = StringIO()
    errors = StringIO()
    github = FakeGithubClient()
    reviewer_configs: list[dict[str, Any]] = []

    def end_input(_: str) -> str:
        raise EOFError

    def reviewer_factory(**kwargs: Any) -> None:
        reviewer_configs.append(kwargs)

    exit_code = main(
        config=_chat_config(),
        input_fn=end_input,
        output=output,
        error=errors,
        github_factory=lambda _: github,
        model_factory=lambda **_: DeterministicInquiryModel(),
        reviewer_factory=reviewer_factory,
    )

    assert exit_code == 0
    assert output.getvalue().startswith("Review Sheep chatbot ready;")
    assert (
        "What would you like to know or review about pull requests?"
        in output.getvalue()
    )
    assert errors.getvalue() == ""
    assert github.closed is True
    assert len(reviewer_configs) == 1
    assert reviewer_configs[0]["source"].__class__.__name__ == (
        "GitHubPullRequestReader"
    )
    assert "checkout" not in reviewer_configs[0]


def test_chat_treats_a_whitespace_base_url_as_unset() -> None:
    base_urls: list[str | None] = []

    def capture_model_config(**kwargs: Any) -> DeterministicInquiryModel:
        base_urls.append(kwargs["base_url"])
        return DeterministicInquiryModel()

    github = FakeGithubClient()
    exit_code = main(
        config=_chat_config(base_url="   "),
        input_fn=lambda _: "quit",
        output=StringIO(),
        error=StringIO(),
        github_factory=lambda _: github,
        model_factory=capture_model_config,
    )

    assert exit_code == 0
    assert base_urls == [None]
    assert github.closed is True


def test_chat_routes_each_question_through_inquiry_agent_and_langgraph() -> None:
    github = FakeGithubClient()
    model = DeterministicInquiryModel()
    prompts = iter(["Tell me about pull request 42.", "quit"])
    output = StringIO()
    errors = StringIO()

    exit_code = main(
        config=_chat_config(),
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


class RecordingTraceHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.metadata.append(metadata or {})


def test_chat_traces_each_user_turn_as_one_langgraph_session() -> None:
    handler = RecordingTraceHandler()
    prompts = iter(["Who are you?", "quit"])

    exit_code = main(
        config=_chat_config(
            base_url="https://injected.example",
        ),
        input_fn=lambda _: next(prompts),
        output=StringIO(),
        error=StringIO(),
        github_factory=lambda _: FakeGithubClient(),
        model_factory=lambda **_: DeterministicInquiryModel(),
        tracing_factory=lambda config: handler,
        tracing_flush=lambda **_: None,
    )

    assert exit_code == 0
    traced = [row for row in handler.metadata if "langfuse_session_id" in row]
    assert len(traced) > 1
    assert {row["review_sheep_turn"] for row in traced} == {1}
    assert len({row["langfuse_session_id"] for row in traced}) == 1
    assert all("review-sheep" in row["langfuse_tags"] for row in traced)


def test_chat_uses_explicit_config_over_environment(monkeypatch: Any) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "environment-token")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    captured: dict[str, Any] = {}
    langfuse = LangfuseConfig(
        public_key="pk-injected",
        secret_key="sk-injected",
        base_url="https://trace-injected.example",
        environment="test",
    )

    def github_factory(token: str) -> FakeGithubClient:
        captured["github_token"] = token
        return FakeGithubClient()

    def model_factory(**kwargs: Any) -> DeterministicInquiryModel:
        captured.update(kwargs)
        return DeterministicInquiryModel()

    def tracing_factory(config: LangfuseConfig | None) -> object:
        captured["tracing"] = config
        return object()

    exit_code = main(
        config=ChatConfig(
            github_token="injected-token",
            model="injected-model",
            api_key="injected-key",
            base_url="https://model-injected.example",
            langfuse=langfuse,
        ),
        input_fn=lambda _: "quit",
        output=StringIO(),
        error=StringIO(),
        github_factory=github_factory,
        model_factory=model_factory,
        tracing_factory=tracing_factory,
        tracing_flush=lambda **_: None,
    )

    assert exit_code == 0
    assert captured == {
        "github_token": "injected-token",
        "model": "injected-model",
        "api_key": "injected-key",
        "base_url": "https://model-injected.example",
        "tracing": langfuse,
    }


def test_chat_redacts_credentials_from_error_output() -> None:
    config = ChatConfig(
        github_token="github-secret-injected",
        model="gpt-test",
        api_key="model-secret-injected",
    )
    errors = StringIO()

    def unavailable_github(token: str) -> Any:
        raise RuntimeError(f"token={token} api_key={config.api_key}")

    exit_code = main(
        config=config,
        input_fn=lambda _: "quit",
        output=StringIO(),
        error=errors,
        github_factory=unavailable_github,
    )

    assert exit_code == 1
    assert "github-secret-injected" not in errors.getvalue()
    assert "model-secret-injected" not in errors.getvalue()
    assert "[redacted]" in errors.getvalue()
