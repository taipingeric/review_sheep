from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from llm import build_llm
from tools import github_toolkit, review_code

load_dotenv()

SYSTEM_PROMPT = (
    "You are a repository assistant, ask user for the repository name under langchain-ai/langgraph if it is not specified. "
    "Reporting rules:\n"
    "- Cite PR numbers and URLs so the user can open them.\n"
    "- If a result has truncated=true, say the list is incomplete rather "
    "than implying it is the full set.\n"
    "- Report only what the tools return. Do not guess at PR contents, "
    "review status, or file names.\n"
    "- Reply in the language the user writes in."
)


def build_agent():
    """Build the agent; create_agent handles the model and tool-call loop."""
    return create_agent(
        model=build_llm(),
        tools=[review_code, *github_toolkit],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def main() -> None:
    """Chat with the agent from the terminal, streaming the reply."""
    agent = build_agent()
    config = {"configurable": {"thread_id": "cli"}}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        print("Assistant: ", end="", flush=True)
        for message, _ in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(message, AIMessageChunk) and message.text:
                print(message.text, end="", flush=True)
        print()


if __name__ == "__main__":
    main()