# Review Sheep

Review Sheep is a Python library for reading GitHub pull requests through two
separate capabilities:

- **Inquiry** answers lightweight questions from pull-request metadata.
- **Review** asks whole-pull-request Lens agents to inspect either a GitHub API
  Manifest (interactive Chat) or a complete fixed-SHA Git checkout (CI), then
  returns structured Pydantic Findings.

The Review Sheep Python library is read-only. It never submits reviews, changes
labels, or otherwise writes Findings back to GitHub. Publishing a rendered
Report is the caller's responsibility; the bundled CI workflow is one such
caller and publishes the Report as a pull-request conversation comment.

The implementation uses LangChain agents for model and tool execution, and
LangGraph for conversational state orchestration. See
[ARCHITECTURE.md](https://github.com/taipingeric/review_sheep/blob/main/ARCHITECTURE.md)
for the complete design.

## Install

Review Sheep requires Python 3.11 or newer.

```bash
pip install review-sheep
```

For an OpenAI-backed model, install the genuinely optional provider extra:

```bash
pip install "review-sheep[openai]"
```

`anthropic_model` and Langfuse tracing have their own extras too:

```bash
pip install "review-sheep[anthropic]"
pip install "review-sheep[tracing]"
```

`langchain-anthropic` is also pulled in transitively today by the
`deepagents` dependency, so `anthropic_model` works even without the
`anthropic` extra; declaring it keeps Review Sheep's own dependency contract
accurate independent of what `deepagents` requires.

To install from source instead:

```bash
pip install .
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
    create_intent_classifier,
    create_manifest_review_agent,
)

chatbot = create_chatbot_graph(
    agent=inquiry_agent,
    classifier=create_intent_classifier(model=model),
    reviewer=create_manifest_review_agent(source=github, model=model),
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

For CI and library callers with a complete checkout, Review runs a LangGraph
workflow that verifies one clean local checkout against GitHub's PR base/head
SHAs. Correctness, security, and conventions-and-tests agents share that fixed
checkout and run concurrently. They generate diffs from `git diff
base...head` and read complete source files directly. Findings retain their
Location, Severity, Confidence, and originating Lens; results are merged in
stable Lens order and overlapping Findings are not merged.

```python
from review_sheep import (
    GitCheckoutSource,
    Review,
    create_deep_review_agent,
    render_report,
)

checkout = GitCheckoutSource(revisions=github, root="/work/pr-42")
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

## Use Review as a LangChain tool

`review_pull_request_tool` wraps the Manifest Review workflow — fetching a
GitHub API snapshot, running every Lens, and rendering the Report — behind
one LangChain tool, so it can be dropped straight into another agent's own
`tools=[...]` list instead of wiring `GitHubPullRequestReader`,
`create_manifest_review_agent`, and `render_report` together by hand:

```python
from review_sheep import GitHubPullRequestReader, review_pull_request_tool

source = GitHubPullRequestReader(client=github)
tool = review_pull_request_tool(model, source)

my_agent = create_agent(model=model, tools=[*my_tools, tool])
```

The tool takes `repo` and `number` arguments and returns the rendered
Markdown Report as a string; a failed Review is returned as descriptive error
text rather than raised, so the calling agent can reason about or relay it.

## Try the intent-routed chatbot

Install the project with the provider extra you want to test. For OpenAI:

```bash
uv sync --extra openai --extra dev
```

The interactive script reads `GITHUB_TOKEN`, `OPENAI_MODEL`, `OPENAI_API_KEY`,
and optional `BASE_URL` and `REVIEW_LOG_LEVEL` from `.env`:

```dotenv
GITHUB_TOKEN=your-read-only-github-token
OPENAI_MODEL=gpt-5-mini
OPENAI_API_KEY=your-openai-api-key
REVIEW_LOG_LEVEL=INFO
# BASE_URL=https://your-compatible-endpoint/v1
# Optional Langfuse tracing:
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
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
repositories require corresponding read access. For changed-code Review, Chat
fetches the PR files from GitHub, verifies that the head SHA stayed stable, and
creates an in-memory `/manifest.json` plus `/diffs/<path>.diff` files. It does
not clone the target repository and does not require `REVIEW_CHECKOUT`.

Progress logs go to stderr. `INFO` shows intent routing, GitHub snapshot,
Manifest construction, each Lens, Finding counts, and Report rendering. Set
`REVIEW_LOG_LEVEL=DEBUG` to also print Manifest paths, patch sizes, tool results,
and structured Findings. Credentials and complete source/diff contents are not
logged. Both Chat routes remain read-only and never post reviews, comments, or
Findings to GitHub.

Langfuse tracing is optional and automatically activates when both
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present. Each user turn is a
`review-sheep-chat-turn` trace; all turns from one terminal process share a
generated Langfuse session ID. The callback attached to the outer Chat
LangGraph propagates into intent classification, Inquiry, Manifest Review, and
the Lens agents. Set `LANGFUSE_TRACING_ENABLED=false` to explicitly disable it.
The client flushes queued spans before the CLI exits. See the
[Langfuse LangGraph integration](https://langfuse.com/guides/cookbook/integration_langgraph).

Tracing sends prompts, model responses, graph state, and tool activity to the
configured Langfuse project. Use a Langfuse deployment and retention policy
appropriate for the repositories being reviewed.

## GitHub Actions PR Review

The repository provides a reusable workflow that downloads Review Sheep into a
directory separate from the PR checkout, installs it with `uv`, and calls the
non-interactive `scripts/review_pr.py`. The Markdown Report is printed in the
job log, appended to the GitHub Actions step summary, and published as a
pull-request conversation comment. Later runs update the existing Review Sheep
comment instead of creating duplicates.

This repository also contains `.github/workflows/review.yml`, which runs the
reusable workflow automatically for same-repository pull requests. Its preferred
enterprise-style configuration matches the COP/Kiro workflow:

- Actions secret `COP_KIRO_AUTH_TOKEN` contains the gateway bearer token;
- Actions variable `COP_KIRO_BASE_URL` contains the non-secret gateway URL;
- optional Actions secret `COP_KIRO_CUSTOM_HEADERS` contains `Name: Value` lines;
- optional Actions variables `COP_KIRO_HAIKU_MODEL`, `COP_KIRO_OPUS_MODEL`, and
  `COP_KIRO_SONNET_MODEL` override the built-in Claude model IDs; and
- optional Actions variables `REVIEW_MODEL_TIER` (`haiku`, `sonnet`, or `opus`,
  default `sonnet`) and `REVIEW_INSTRUCTIONS` control the Review.

For compatibility, the caller falls back to the existing `ANTHROPIC_*` secrets
and variables. If the gateway URL is sensitive, store it as `COP_KIRO_BASE_URL`
under Actions secrets instead of variables; the workflow accepts either source.

Fork pull requests are intentionally skipped because the `pull_request` event
does not expose repository model secrets to untrusted forks.

Configure the Actions secrets and variables above in the repository being
reviewed, then create `.github/workflows/review-sheep.yml` there:

```yaml
name: Review Sheep

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    # pull_request does not expose repository secrets to fork PRs.
    if: github.event.pull_request.head.repo.full_name == github.repository
    uses: taipingeric/review_sheep/.github/workflows/review-pr.yml@REVIEW_SHEEP_SHA
    with:
      # Use the same full commit SHA as the reusable workflow reference above.
      review_sheep_ref: REVIEW_SHEEP_SHA
      model_tier: sonnet
      anthropic_base_url: ${{ vars.COP_KIRO_BASE_URL }}
      haiku_model: ${{ vars.COP_KIRO_HAIKU_MODEL || 'claude-haiku-4-5' }}
      opus_model: ${{ vars.COP_KIRO_OPUS_MODEL || 'claude-opus-4-6' }}
      sonnet_model: ${{ vars.COP_KIRO_SONNET_MODEL || 'claude-sonnet-4-6' }}
      # instructions: Focus on authorization and data integrity.
    secrets:
      ANTHROPIC_AUTH_TOKEN: ${{ secrets.COP_KIRO_AUTH_TOKEN }}
      ANTHROPIC_BASE_URL: ${{ secrets.COP_KIRO_BASE_URL }}
      ANTHROPIC_CUSTOM_HEADERS: ${{ secrets.COP_KIRO_CUSTOM_HEADERS }}
```

Replace both `REVIEW_SHEEP_SHA` placeholders with the same full Review Sheep
commit SHA. Pinning prevents workflow code and the downloaded Python package
from drifting independently. Do not change this workflow to
`pull_request_target` and then execute untrusted PR code with model secrets.

`ANTHROPIC_AUTH_TOKEN` is passed through the Anthropic SDK's bearer-token
authentication path. `ANTHROPIC_CUSTOM_HEADERS` follows Claude Code's
newline-separated `Name: Value` convention.

Gateway connections use a 10-second connect timeout with the Anthropic SDK's
normal retries, while model responses may run for up to 10 minutes. A
`ConnectTimeout` therefore indicates that the gateway cannot be reached from
the GitHub runner; it is not a model-generation timeout. Private gateways need
a self-hosted runner with network access or an endpoint reachable from GitHub's
hosted runners.

The reusable workflow checks out the exact PR head with full Git history, so
`GitCheckoutSource` can verify the head SHA and resolve the base commit before
any Lens runs.

## Publishing to PyPI

`.github/workflows/publish.yml` builds the sdist and wheel with `uv build` and
uploads them to PyPI whenever a GitHub Release is published, or a `v*` tag is
pushed. It authenticates with [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC), so no PyPI API token is stored as a repository secret.

Before the first release, a repository maintainer must sign in to pypi.org and
add a pending publisher for the `review-sheep` project pointing at
`taipingeric/review_sheep` and the `publish.yml` workflow. This is a one-time,
manual step tied to a PyPI account and cannot be automated.
