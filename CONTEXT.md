# Review Sheep

A Python library for building agents that review GitHub pull requests. It reads
pull requests through the GitHub API and produces structured findings.

The implementation is based on LangChain agents and LangGraph orchestration.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete runtime architecture.

## Language

**Finding**:
A single problem identified in a pull request, anchored to a location in the
changed code and carrying a severity. The atomic unit of a Review's output.
_Avoid_: comment, issue, suggestion, remark, nit

**Review**:
One pass over a single pull request that produces a set of Findings.
_Avoid_: audit, analysis, scan, check

**Report**:
A rendered, human-facing presentation of a Review's Findings. Always derived
from Findings, never the source of truth.
_Avoid_: summary, output, feedback, result

**Severity**:
How bad a Finding's consequence is if the Finding is true: `critical` (data
loss, exploitable vulnerability, or production outage), `high` (wrong behaviour
or a crash on foreseeable input), `medium` (fails only in edge cases, or
materially burdens maintenance), `low` (convention and readability, no
behavioural impact).
_Avoid_: priority, importance, level

**Confidence**:
How sure a Finding is real, given that a reviewer sees only the code it was
shown: `confirmed`, `likely`, or `speculative`. Independent of Severity — a
Finding can be critical-if-true yet speculative.
_Avoid_: certainty, probability, score

**Location**:
Where a Finding sits: a path in the pull request's head commit, optionally
narrowed to a line range. A Finding with no line range is about the file as a
whole, or about the pull request as a whole when it has no path either.
_Avoid_: position, anchor, span, coordinates

**Lens**:
One concern a Review examines the whole pull request through, such as
correctness, security, or conventions and tests. Each Lens is reviewed
independently and contributes Findings to the Review.
_Avoid_: aspect, dimension, category, pass

**Manifest**:
The index of a pull request's changed files — path, change status, and line
counts — written alongside the fetched diffs so a Lens can decide which files to
read.
_Avoid_: index, listing, file list, inventory

**Inquiry**:
A question about pull requests answered from GitHub metadata alone, producing no
Findings. Distinct from a Review, which examines changed code.
_Avoid_: chat, query, quick mode, ask
