# Review Sheep

Review Sheep is a Python library for reading GitHub pull requests through two
separate capabilities:

- **Inquiry** answers lightweight questions from pull-request metadata.
- **Review** fetches one stable changed-code snapshot, asks whole-pull-request
  Lens subagents to inspect it, and returns structured Pydantic Findings.

Review Sheep is read-only. It never submits reviews, posts comments, changes
labels, or otherwise writes Findings back to GitHub. Publishing a rendered
Report is the caller's responsibility.

The implementation uses LangChain agents for model and tool execution, and
LangGraph for conversational state orchestration. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the complete design.

## Install

Review Sheep requires Python 3.11 or newer.

```bash
pip install .
```

For an OpenAI-backed model, install the genuinely optional provider extra:

```bash
pip install ".[openai]"
```

## Inquiry

The caller constructs the model and GitHub client explicitly. Inquiry exposes
metadata-only tools; it does not fetch changed code or run Review subagents.

```python
from github import Auth, Github
from langchain_openai import ChatOpenAI

from review_sheep import GitHubPullRequestReader, create_inquiry_agent

github = GitHubPullRequestReader(
    client=Github(auth=Auth.Token("github-token")),
    default_repo="acme/widgets",
)
model = ChatOpenAI(model="gpt-5-mini", api_key="openai-key")

inquiry_agent = create_inquiry_agent(model=model, github=github)
answer = inquiry_agent.ask("Which pull requests are awaiting review?")
if answer.error:
    print(answer.error)
else:
    print(answer.text)
```

## Simple Chatbot Graph

Build a continuous LangGraph chatbot around the LangChain Inquiry agent. Each
turn uses exactly one node and follows `START -> Bot -> END`; LangGraph keeps the
conversation messages while the Inquiry agent selects read-only GitHub metadata
tools and answers each question:

```python
from langchain_core.messages import HumanMessage

from review_sheep import InquiryChatState, create_chatbot_graph

chatbot = create_chatbot_graph(agent=inquiry_agent)
state = InquiryChatState(messages=[])
state = chatbot.invoke(state)
print(state["messages"][-1].content)

# Continue by appending each user reply to the returned state.
state["messages"] = [
    *state["messages"],
    HumanMessage(content="Which open PRs in acme/widgets need review?"),
]
state = chatbot.invoke(state)
```

The `Bot` node sends the accumulated conversation to `InquiryAgent`. Completed
graph output keeps the structured `InquiryAnswer` in `state["answer"]` and
appends its text as the final AI message.

## Review and Report

Review reads changed-file metadata and diffs once. Correctness, security, and
conventions-and-tests subagents inspect the same in-memory Manifest and diff
snapshot. Findings retain their Location, Severity, Confidence, and originating
Lens; overlapping Findings are not merged.

```python
from review_sheep import (
    Review,
    create_deep_review_agent,
    render_report,
)

reviewer = create_deep_review_agent(source=github, model=model)
result = reviewer.review(repo="acme/widgets", number=42)

if isinstance(result, Review):
    report = render_report(result)
    print(report.text)
else:
    print(
        f"{result.operation.value} failed for "
        f"{result.repo}#{result.pull_request_number}: {result.message}"
    )
```

The lower-level `create_review_agent(source=..., runner=...)` seam accepts
deterministic collaborators for tests or a caller-owned Review implementation.

## Try live Inquiry prompts in a loop

Install the project with the provider extra you want to test. For OpenAI:

```bash
uv sync --extra openai --extra dev
```

The interactive script reads `GITHUB_TOKEN`, `OPENAI_MODEL`, `OPENAI_API_KEY`,
and optional `BASE_URL` from `.env`:

```dotenv
GITHUB_TOKEN=your-read-only-github-token
OPENAI_MODEL=gpt-5-mini
OPENAI_API_KEY=your-openai-api-key
# BASE_URL=https://your-compatible-endpoint/v1
```

```bash
uv run python scripts/review_chat.py
```

Ask metadata questions naturally. The Inquiry agent can list pull requests, read
one pull request, and inspect its review state. Type `exit` or `quit` to stop.

```text
Inquiry chatbot ready; type exit or quit to stop.
Bot: What would you like to know about pull requests?
You: Which pull requests are open in other/project?
Bot: Open pull requests: #123 ...
You: What is the review state of #123?
Bot: Pull request #123 ...
You: quit
```

The GitHub token must be able to read the target repository; private
repositories require corresponding read access. Inquiry is metadata-only and
never posts reviews, comments, or Findings to GitHub. Changed-code Review is
available separately through `create_deep_review_agent`.
