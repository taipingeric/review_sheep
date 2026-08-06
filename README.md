# Review Sheep

Review Sheep is a Python library for reading GitHub pull requests through two
separate capabilities:

- **Inquiry** answers lightweight questions from pull-request metadata.
- **Review** fetches one stable changed-code snapshot, asks whole-pull-request
  Lens subagents to inspect it, and returns structured Pydantic Findings.

Review Sheep is read-only. It never submits reviews, posts comments, changes
labels, or otherwise writes Findings back to GitHub. Publishing a rendered
Report is the caller's responsibility.

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

from review_sheep import GitHubPullRequestReader, create_inquiry

github = GitHubPullRequestReader(
    client=Github(auth=Auth.Token("github-token")),
    default_repo="acme/widgets",
)
model = ChatOpenAI(model="gpt-5-mini", api_key="openai-key")

answer = create_inquiry(model=model, github=github).ask(
    "Which pull requests are awaiting review?"
)
if answer.error:
    print(answer.error)
else:
    print(answer.text)
```

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
