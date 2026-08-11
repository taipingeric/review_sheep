# Review Sheep

Review Sheep is a Python library for reading GitHub pull requests through two
separate capabilities:

- **Inquiry** answers lightweight questions from pull-request metadata.
- **Review** reads a complete local Git checkout pinned to a pull-request head
  SHA, asks whole-pull-request Lens agents to inspect it, and returns structured
  Pydantic Findings.

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
    GitCheckoutSource,
    create_chatbot_graph,
    create_deep_review_agent,
    create_intent_classifier,
)

checkout = GitCheckoutSource(revisions=github, root="/work/pr-42")

chatbot = create_chatbot_graph(
    agent=inquiry_agent,
    classifier=create_intent_classifier(model=model),
    reviewer=create_deep_review_agent(source=checkout, model=model),
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

Review runs a LangGraph workflow that verifies one clean local checkout against
GitHub's PR base/head SHAs and invokes the Lens agents. Correctness, security,
and conventions-and-tests agents share that fixed checkout. They generate the
changed-file index and diffs from `git diff base...head`, then read complete
source files directly. No `manifest.json` is generated. Findings retain their
Location, Severity, Confidence, and originating Lens; overlapping Findings are
not merged.

```python
from review_sheep import (
    Review,
    create_deep_review_agent,
    render_report,
)

reviewer = create_deep_review_agent(source=checkout, model=model)
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
and optional `BASE_URL` and `REVIEW_CHECKOUT` from `.env`:

```dotenv
GITHUB_TOKEN=your-read-only-github-token
OPENAI_MODEL=gpt-5-mini
OPENAI_API_KEY=your-openai-api-key
REVIEW_CHECKOUT=/absolute/path/to/a/clean/pr-checkout
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
repositories require corresponding read access. Before a changed-code Review,
the configured checkout must have the PR head SHA at `HEAD`, contain the PR base
commit, and have no tracked or untracked changes. CI can prepare that checkout
before starting the agent. Both routes are read-only and never post reviews,
comments, or Findings to GitHub.

## GitHub Actions PR Review

The repository provides a reusable workflow that downloads Review Sheep into a
directory separate from the PR checkout, installs it with `uv`, and calls the
non-interactive `scripts/review_pr.py`. The Markdown Report is printed in the
job log and appended to the GitHub Actions step summary; it is not posted to the
pull request.

This repository also contains `.github/workflows/review.yml`, which runs the
reusable workflow automatically for same-repository pull requests. Configure
these repository **Actions secrets**:

- `ANTHROPIC_AUTH_TOKEN` (required bearer token);
- `ANTHROPIC_BASE_URL` (required gateway URL);
- `ANTHROPIC_CUSTOM_HEADERS` (optional `Name: Value` lines);
- `ANTHROPIC_DEFAULT_HAIKU_MODEL`;
- `ANTHROPIC_DEFAULT_OPUS_MODEL`; and
- `ANTHROPIC_DEFAULT_SONNET_MODEL` (the default Review model).

Optional Actions variables are `REVIEW_MODEL_TIER` (`haiku`, `sonnet`, or
`opus`, default `sonnet`) and `REVIEW_INSTRUCTIONS`.

Fork pull requests are intentionally skipped because the `pull_request` event
does not expose repository model secrets to untrusted forks.

Add the Anthropic Actions secrets above to the repository being reviewed, then
create `.github/workflows/review-sheep.yml` there:

```yaml
name: Review Sheep

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: read

jobs:
  review:
    # pull_request does not expose repository secrets to fork PRs.
    if: github.event.pull_request.head.repo.full_name == github.repository
    uses: taipingeric/review_sheep/.github/workflows/review-pr.yml@REVIEW_SHEEP_SHA
    with:
      # Use the same full commit SHA as the reusable workflow reference above.
      review_sheep_ref: REVIEW_SHEEP_SHA
      model_tier: sonnet
      # instructions: Focus on authorization and data integrity.
    secrets:
      ANTHROPIC_AUTH_TOKEN: ${{ secrets.ANTHROPIC_AUTH_TOKEN }}
      ANTHROPIC_BASE_URL: ${{ secrets.ANTHROPIC_BASE_URL }}
      ANTHROPIC_CUSTOM_HEADERS: ${{ secrets.ANTHROPIC_CUSTOM_HEADERS }}
      ANTHROPIC_DEFAULT_HAIKU_MODEL: ${{ secrets.ANTHROPIC_DEFAULT_HAIKU_MODEL }}
      ANTHROPIC_DEFAULT_OPUS_MODEL: ${{ secrets.ANTHROPIC_DEFAULT_OPUS_MODEL }}
      ANTHROPIC_DEFAULT_SONNET_MODEL: ${{ secrets.ANTHROPIC_DEFAULT_SONNET_MODEL }}
```

Replace both `REVIEW_SHEEP_SHA` placeholders with the same full Review Sheep
commit SHA. Pinning prevents workflow code and the downloaded Python package
from drifting independently. Do not change this workflow to
`pull_request_target` and then execute untrusted PR code with model secrets.

`ANTHROPIC_AUTH_TOKEN` is passed through the Anthropic SDK's bearer-token
authentication path. `ANTHROPIC_CUSTOM_HEADERS` follows Claude Code's
newline-separated `Name: Value` convention.

The reusable workflow checks out the exact PR head with full Git history, so
`GitCheckoutSource` can verify the head SHA and resolve the base commit before
any Lens runs.
