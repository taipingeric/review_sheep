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

Build a continuous LangGraph chatbot that classifies each turn and routes it to
the LangChain Inquiry agent, changed-code Review agent, or an unrelated-message
scope response:

```text
START -> IntentClassifier -> Bot | ReviewBot | UnrelatedBot -> END
```

```python
from langchain_core.messages import HumanMessage

from review_sheep import (
    ChatState,
    create_chatbot_graph,
    create_deep_review_agent,
    create_intent_classifier,
)

chatbot = create_chatbot_graph(
    agent=inquiry_agent,
    classifier=create_intent_classifier(model=model),
    reviewer=create_deep_review_agent(source=github, model=model),
)
state = ChatState(messages=[])
state = chatbot.invoke(state)
print(state["messages"][-1].content)

# Continue by appending each user reply to the returned state.
state["messages"] = [
    *state["messages"],
    HumanMessage(content="Which open PRs in acme/widgets need review?"),
]
state = chatbot.invoke(state)
```

`IntentClassifier` returns a Pydantic routing decision. `Bot` keeps the
structured `InquiryAnswer` in `state["answer"]`; `ReviewBot` keeps the
structured `Review` or `ReviewError` in `state["review"]`. `UnrelatedBot` does
not invoke either agent and returns only the chatbot's supported scope.

## Review and Report

Review runs a LangGraph workflow that reads changed-file metadata and diffs
once, prepares a shared workspace, and invokes the Lens agents. Correctness,
security, and conventions-and-tests deep agents inspect the same in-memory
Manifest and diff snapshot. Findings retain their Location, Severity,
Confidence, and originating Lens; overlapping Findings are not merged.

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

## Try the intent-routed chatbot

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

Ask what Review Sheep can do, ask metadata questions, or request a changed-code
Review naturally. The classifier selects the correct agent path; ReviewBot asks
for a repository or pull-request number when the request omitted it. Type `exit`
or `quit` to stop.

```text
Review Sheep chatbot ready; type exit or quit to stop.
Bot: What would you like to know or review about pull requests?
You: Which pull requests are open in other/project?
Bot: Open pull requests: #123 ...
You: What is the review state of #123?
Bot: Pull request #123 ...
You: Review changed code in other/project#123 for correctness
Bot: # Review Report: other/project#123
You: quit
```

The GitHub token must be able to read the target repository; private
repositories require corresponding read access. Both routes are read-only and
never post reviews, comments, or Findings to GitHub.
