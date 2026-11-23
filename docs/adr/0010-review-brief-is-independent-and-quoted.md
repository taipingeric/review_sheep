# The Review Brief is independent of the Findings and quotes its sources

Status: accepted

With `brief=True` (the default for the fixed-checkout Review), one extra agent
writes a Review Brief: Intent, Acceptance Criteria with verdicts, and Decisions.
It runs in parallel with the Lenses and never sees their Findings.

Independence keeps a `met` verdict a second opinion rather than a restatement of
the Lenses. Feeding the Brief to the Lenses, or updating it after the Findings,
would anchor them on it and cost one more call.

An Acceptance Criterion must come from a document: the pull request description
or a Review Context. Each one names its source and quotes the exact words, and
code drops any criterion whose quote is not in that named source. Without this
the agent would write the exam and grade it, and a `met` would mean nothing.
Decisions work the same way: the agent searches the repo for recorded decisions
(ADRs, RFCs) first. A relevant record becomes a Decision with `source` set to
its path and the exact quote; code drops it if the quote is not in that file.
Only when no record applies does the agent write one, with `source: generated`.
Evidence lines are checked the same way: lines that do not exist in the checkout
are dropped, and a `met` left with no real evidence becomes `unverified`.

## Consequences

A Review costs one more agent run, and the Brief is empty for a pull request
whose documents state no criteria; the Report says so. The checks prove that the
quoted words and cited lines exist, not that they support the verdict, and an
agent can still misread a real quote. A failed Brief never fails the Review: it
stays empty and `brief_error` records why. The Lenses fail the Review, so when
they fail the Brief's agent run is wasted. The Manifest Review used by Chat has
no checkout to read and writes no Brief.
