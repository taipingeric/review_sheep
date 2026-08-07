from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from review_sheep import (
    ChatIntent,
    IntentDecision,
    create_intent_classifier,
)


class StructuredIntentModel(BaseChatModel):
    tool_name: str | None = None
    route_intent: str = "review"

    @property
    def _llm_type(self) -> str:
        return "structured-intent-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> StructuredIntentModel:
        return self.model_copy(update={"tool_name": tools[0].name})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        assert self.tool_name is not None
        args: dict[str, Any] = {"intent": self.route_intent}
        if self.route_intent == "review":
            args.update(
                {
                    "repo": "acme/widgets",
                    "pull_request_number": 42,
                }
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": self.tool_name,
                                "args": args,
                                "id": "call-intent",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )


class FailingIntentModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing-intent-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FailingIntentModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("classifier unavailable")


def test_intent_classifier_returns_structured_review_route() -> None:
    classifier = create_intent_classifier(model=StructuredIntentModel())

    decision = classifier.classify(
        [HumanMessage(content="Review correctness in acme/widgets pull request 42.")]
    )

    assert decision == IntentDecision(
        intent=ChatIntent.REVIEW,
        repo="acme/widgets",
        pull_request_number=42,
    )


def test_intent_classifier_returns_unrelated_for_an_out_of_scope_message() -> None:
    classifier = create_intent_classifier(
        model=StructuredIntentModel(route_intent="unrelated")
    )

    decision = classifier.classify([HumanMessage(content="What is the weather?")])

    assert decision == IntentDecision(intent=ChatIntent.UNRELATED)


def test_intent_classifier_routes_identity_questions_to_inquiry() -> None:
    classifier = create_intent_classifier(
        model=StructuredIntentModel(route_intent="inquiry")
    )

    decision = classifier.classify([HumanMessage(content="Who are you?")])

    assert decision == IntentDecision(intent=ChatIntent.INQUIRY)


def test_intent_classifier_contains_model_failure_as_unknown_route() -> None:
    classifier = create_intent_classifier(model=FailingIntentModel())

    decision = classifier.classify(
        [HumanMessage(content="Which pull requests are open?")]
    )

    assert decision == IntentDecision(
        intent=ChatIntent.UNKNOWN,
        error="RuntimeError: classifier unavailable",
    )
