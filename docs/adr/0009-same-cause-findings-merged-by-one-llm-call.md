# Same-cause Findings are merged by one LLM call

Status: supersedes ADR-0005

After every Lens finishes, one extra structured LLM call groups Findings that
share a root cause. Each group becomes one Finding: it keeps the text, Location,
and Confidence of its most severe member (the earliest on a tie), and its
`lenses` field lists every Lens that reported it. `lens` stays the Lens whose
text was kept.

Location alone cannot find duplicates. The same missing permission check shows
up at the handler for one Lens, at the caller for another, and at the missing
test for a third, so only a semantic judgement can join them.

## Consequences

A Review costs one more model call when it has two or more Findings. The model
only returns index groups; the merge itself is deterministic code, so bad
indices are dropped and every Finding still appears exactly once. If the call
fails, the Review logs it and returns the Findings unmerged instead of failing.
