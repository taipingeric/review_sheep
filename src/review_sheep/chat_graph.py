"""A conversational LangGraph chatbot backed by the Inquiry agent."""

from __future__ import annotations

from typing import NotRequired

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from review_sheep.domain import InquiryAnswer
from review_sheep.inquiry import InquiryAgent


class InquiryChatState(MessagesState):
    """Conversation messages plus the latest structured Inquiry answer."""

    answer: NotRequired[InquiryAnswer]


def create_chatbot_graph(
    *,
    agent: InquiryAgent,
) -> CompiledStateGraph[
    InquiryChatState,
    None,
    InquiryChatState,
    InquiryChatState,
]:
    """Build a ``START -> Bot -> END`` graph around an Inquiry agent."""

    def bot(state: InquiryChatState) -> dict[str, object]:
        if not state["messages"]:
            return _reply("What would you like to know about pull requests?")

        answer = agent.invoke(state["messages"])
        response = answer.text or f"Inquiry failed: {answer.error}"
        return {
            "messages": [AIMessage(content=response)],
            "answer": answer,
        }

    graph = StateGraph(InquiryChatState)
    graph.add_node("Bot", bot)
    graph.add_edge(START, "Bot")
    graph.add_edge("Bot", END)
    return graph.compile()


def _reply(message: str, **state: object) -> dict[str, object]:
    return {"messages": [AIMessage(content=message)], **state}
