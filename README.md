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

## Test coverage

Run the full test suite with per-module and total statement coverage using the
development dependency set:

```bash
uv run --extra dev pytest --cov=src/review_sheep --cov-report=term-missing
```

The command exits non-zero when tests fail and does not require a manual
coverage-tool installation. Before this coverage check was added, the baseline
was 104 tests with 94% statement coverage across 1,218 statements, with 78
statements missed. Coverage is reported for the `review_sheep` source package;
generated coverage files remain ignored by Git.

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

For repeated Reviews of the same repository, use `CachedGitCheckoutSource` to
keep one credential-free bare mirror while creating a fresh disposable worktree
for every PR revision. Git authentication remains delegated to Git's normal
credential helper or environment; do not put tokens in `remote_url`.

```python
from review_sheep import CachedGitCheckoutSource, create_deep_review_agent

checkout = CachedGitCheckoutSource(
    revisions=github,
    cache_root="~/.cache/review-sheep/repos",
)
reviewer = create_deep_review_agent(source=checkout, model=model)
result = reviewer.review(repo="acme/widgets", number=42)
```

The Review graph removes the disposable worktree after both successful and
failed Reviews; the bare mirror remains available for the next Review.
Mirror mutation is coordinated by a lock per repository, so different
repositories can proceed independently. `lock_timeout` defaults to 30 seconds;
contention beyond that raises `RepositoryMirrorLockTimeout` and becomes a
structured `prepare_checkout` `ReviewError` when used through the Review graph.
Call `cleanup_stale()` periodically to remove worktrees unused for 24 hours
and mirrors unused for 30 days. It returns counts for removed resources,
active resources skipped, and actionable cleanup errors; active worktrees hold
leases and are never removed by that pass. Failed initialization or fetches are
retried through a safe mirror rebuild.

`create_deep_review_agent` also accepts an ordered collection of custom Lenses.
They run concurrently with the built-in Lenses over the same fixed checkout and
return standard Findings tagged with their custom identifiers:

```python
from review_sheep import CustomLens, create_deep_review_agent

reviewer = create_deep_review_agent(
    source=checkout,
    model=model,
    custom_lenses=[
        CustomLens(
            identifier="api-compatibility",
            instructions="Review public API changes for compatibility regressions.",
        )
    ],
)
```

### Define a custom Lens

Review extensions can describe a caller-owned review perspective with the
public `CustomLens` contract. Each custom Lens has a stable lower-case,
kebab-case `identifier` and non-blank `instructions`. Identifiers cannot reuse
the built-in `correctness`, `security`, or `conventions-and-tests` names, and
`validate_custom_lenses` rejects duplicate identifiers while preserving caller
order. Findings can carry either a built-in `Lens` value or a custom identifier;
built-in Finding serialization remains unchanged. The `instructions` value is
the custom Lens's caller-owned, concern-specific review guidance; it does not
change the built-in Lens definitions or the Review's shared snapshot contract.

Pass the ordered definitions to `create_manifest_review_agent` or
`review_pull_request_tool` to run them alongside the built-in Lenses. All
Lenses receive the same Manifest snapshot, virtual diff files, and caller-owned
tools; Findings are returned in built-in order followed by custom declaration
order:

```python
custom_lenses = [
    CustomLens(
        identifier="api-compatibility",
        instructions="Review public API changes for compatibility regressions.",
    )
]
reviewer = create_manifest_review_agent(
    source=source,
    model=model,
    custom_lenses=custom_lenses,
)
```

### Add caller-provided Review tools

Both Review constructors accept optional LangChain tools. Review Sheep gives
each Lens the built-in repository or Manifest tools plus the caller-provided
tools, so a Review can use read-only context such as database schema or table
metadata:

```python
from langchain_core.tools import tool

@tool
def fetch_db_metadata(table: str) -> str:
    """Fetch read-only metadata for one database table."""
    return metadata_client.describe_table(table)

reviewer = create_deep_review_agent(
    source=checkout,
    model=model,
    tools=[fetch_db_metadata],
)
```

`create_manifest_review_agent` and `review_pull_request_tool` accept the same
`tools` argument. Tools are owned and authorized by the caller; Review Sheep
does not manage their credentials or enforce that external operations are
read-only. Prefer narrowly scoped access, bounded results, and thread-safe
implementations because all Lenses run concurrently. Tool results are sent to
the model and can provide evidence for Findings. Custom Lens definitions do not
carry credentials themselves; inject credentials only through caller-owned tools
with the authorization and secret-handling policy appropriate for your
application.

Custom Lens output is validated against the standard `Finding` fields and is
returned unmerged after the built-in Findings, in the declaration order supplied
to `custom_lenses`. The same identity appears in the `Review`, rendered Report,
logs, and tracing metadata. If a Lens fails, the Review returns its existing
`ReviewError` boundary with the failing Lens identity; successful partial results
are exposed only where the surrounding public API already supports partial
results.

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

### Configuration and secret injection

The library entrypoints consume explicit configuration objects. Construct a
`ChatConfig` or `CIConfig` and pass it to `review_sheep.chat.main` or
`review_sheep.ci.main` when embedding Review Sheep in an application. These
core paths do not read process environment values or files.

For example, an application can inject values from its own secret manager:

```python
from review_sheep import ChatConfig
from review_sheep.chat import main

config = ChatConfig(
    github_token=get_secret("GITHUB_TOKEN"),
    model=get_setting("OPENAI_MODEL"),
    api_key=get_secret("OPENAI_API_KEY"),
)
main(config=config)
```

The bundled scripts are process-environment adapters. The interactive script
reads `GITHUB_TOKEN`, `OPENAI_MODEL`, `OPENAI_API_KEY`, and optional `BASE_URL`
and `REVIEW_LOG_LEVEL` from the environment of the process that launches it:

```bash
export GITHUB_TOKEN="<read-only-github-token>"
export OPENAI_MODEL="<model-id>"
export OPENAI_API_KEY="<provider-api-key>"
export REVIEW_LOG_LEVEL="INFO"
# Optional: export BASE_URL="<compatible-endpoint>/v1"
uv run python scripts/review_chat.py
```

The CI adapter uses the environment and command-line values documented in
[GitHub Actions PR Review](#github-actions-pr-review). Its non-secret CLI
arguments (`--repo`, `--pr-number`, `--checkout`, `--instructions`, and
`--model-tier`) take precedence over their environment defaults. Explicit
configuration passed directly to a library entrypoint takes precedence over
all adapter inputs because the core entrypoint does not consult the
environment. Keep GitHub and model credentials in process environment values,
CI secret variables, or a secrets manager; never pass secrets as CLI
arguments, commit them to a checkout, or put them in example files.

Neither production entrypoint loads a local `.env` file. A local shell, CI
secret adapter, or secrets manager must populate the process environment
explicitly. The examples use placeholders only and do not require a checked-out
`.env` file.

The following optional tracing variables are read from the process environment:

- Langfuse: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, optional
  `LANGFUSE_BASE_URL`, `LANGFUSE_TRACING_ENVIRONMENT`, and
  `LANGFUSE_TRACING_ENABLED=false`.
- MLflow: `MLFLOW_TRACKING_URI` and optional `MLFLOW_EXPERIMENT_NAME`.

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

### Optional multi-backend tracing

Tracing is opt-in and each backend is configured independently:

- Langfuse activates when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are
  both present. `LANGFUSE_BASE_URL` and `LANGFUSE_TRACING_ENVIRONMENT` are
  optional; `LANGFUSE_TRACING_ENABLED=false` disables it explicitly.
- MLflow activates when `MLFLOW_TRACKING_URI` is present. The optional
  `MLFLOW_EXPERIMENT_NAME` defaults to `review-sheep`.
- Set both groups to send the same Chat Review turn to both backends. A backend
  initialization failure is logged and does not disable the other backend.

Each Chat user turn is a `review-sheep-chat-turn` trace. The outer Chat
LangGraph callback propagates into intent classification, Inquiry, Manifest
Review, and all Lens agents. The CLI flushes configured backends before exit;
flush failures are logged without replacing the Chat result. See the
[Langfuse LangGraph integration](https://langfuse.com/guides/cookbook/integration_langgraph).

The CI entrypoint uses the same environment variables. Its root trace is named
`review-sheep-ci-review` and uses the stable correlation session
`ci:<repo>#<number>`, including for parallel checkout Lenses. For the reusable
GitHub Actions workflow, pass the optional tracing values through the
`LANGFUSE_*` and `MLFLOW_*` workflow-call secrets shown below.

Tracing callbacks can observe prompts, model responses, LangGraph state, tool
names, tool arguments/results, repository and pull-request metadata, and the
source or diff content included in model/tool activity. Manifest Review sends
virtual manifest/diff data; CI Review can send file and diff content read from
the fixed checkout. Review Sheep does not redact repository content before the
callback receives it, so do not enable tracing for repositories or prompts that
the configured backend must not store.

The same callback events are offered to both adapters, but each backend
serializes them using its own schema: Langfuse receives LangChain/LangGraph
observations with the configured session, tags, run names, prompts, outputs,
and tool observations; MLflow receives corresponding LangChain spans under the
configured tracking URI and experiment. Model metadata such as provider/model
names and invocation metadata may also be included by the LangChain tracer.
The exact fields depend on the installed backend SDK version, so treat both
destinations as capable of receiving the full categories above rather than
assuming that a prompt or diff is private merely because it is not in a
top-level Review result.

Treat the Langfuse project and MLflow tracking server/experiment as sensitive
data destinations: restrict access to the review team, choose retention and
regional policies appropriate for the repositories, and use a self-hosted or
isolated deployment when required by policy. Backend retention and deletion are
controlled by those services; disable tracing or remove the credentials when
the review data must remain offline. Tracing metadata is correlation aid, not
an authorization boundary.

## GitHub Actions PR Review

The repository provides a reusable workflow that downloads Review Sheep into a
directory separate from the PR checkout, installs it with `uv`, and calls the
non-interactive `scripts/review_pr.py`. The Markdown Report is printed in the
job log, appended to the GitHub Actions step summary, and published as a
pull-request conversation comment. Later runs update the existing Review Sheep
comment instead of creating duplicates.

The CI entrypoint receives an explicit `CIConfig`. The script is only an
adapter from trusted environment values and command-line inputs; the Review
path itself does not load a local `.env` file. This lets GitHub Actions and
other CI callers inject credentials and provider settings without placing
secrets in the checkout.

This repository also contains `.github/workflows/review.yml`, which runs the
reusable workflow automatically for same-repository pull requests. Configure
it with:

- Actions secret `ANTHROPIC_AUTH_TOKEN` contains the model provider's bearer
  token;
- Actions variable `ANTHROPIC_BASE_URL` contains the non-secret gateway URL,
  if any;
- optional Actions secret `ANTHROPIC_CUSTOM_HEADERS` contains `Name: Value`
  lines;
- optional Actions variables `ANTHROPIC_DEFAULT_HAIKU_MODEL`,
  `ANTHROPIC_DEFAULT_OPUS_MODEL`, and `ANTHROPIC_DEFAULT_SONNET_MODEL`
  override the built-in Claude model IDs; and
- optional Actions variables `REVIEW_MODEL_TIER` (`haiku`, `sonnet`, or `opus`,
  default `sonnet`) and `REVIEW_INSTRUCTIONS` control the Review.
- optional Actions secrets `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
  `LANGFUSE_BASE_URL`, and `LANGFUSE_TRACING_ENVIRONMENT` configure Langfuse;
- optional Actions secrets `MLFLOW_TRACKING_URI` and
  `MLFLOW_EXPERIMENT_NAME` configure MLflow; and
- set both backend groups to send one CI Review to both tracing systems.

If the gateway URL is sensitive, store it as `ANTHROPIC_BASE_URL` under
Actions secrets instead of variables; the workflow accepts either source.

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
      anthropic_base_url: ${{ vars.ANTHROPIC_BASE_URL }}
      haiku_model: ${{ vars.ANTHROPIC_DEFAULT_HAIKU_MODEL || 'claude-haiku-4-5' }}
      opus_model: ${{ vars.ANTHROPIC_DEFAULT_OPUS_MODEL || 'claude-opus-4-6' }}
      sonnet_model: ${{ vars.ANTHROPIC_DEFAULT_SONNET_MODEL || 'claude-sonnet-4-6' }}
      # instructions: Focus on authorization and data integrity.
    secrets:
      ANTHROPIC_AUTH_TOKEN: ${{ secrets.ANTHROPIC_AUTH_TOKEN }}
      ANTHROPIC_BASE_URL: ${{ secrets.ANTHROPIC_BASE_URL }}
      ANTHROPIC_CUSTOM_HEADERS: ${{ secrets.ANTHROPIC_CUSTOM_HEADERS }}
      LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
      LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
      LANGFUSE_BASE_URL: ${{ secrets.LANGFUSE_BASE_URL }}
      LANGFUSE_TRACING_ENVIRONMENT: ${{ secrets.LANGFUSE_TRACING_ENVIRONMENT }}
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      MLFLOW_EXPERIMENT_NAME: ${{ secrets.MLFLOW_EXPERIMENT_NAME }}
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
