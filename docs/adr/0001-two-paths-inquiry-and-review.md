# Two agent paths: Inquiry and Review

Answering "which PRs are open?" and reviewing a 40-file diff have very different
cost and capability needs, so we build two paths rather than one: an Inquiry path
on a plain tool-calling agent over read-only GitHub metadata, and a Review path
on a deep agent with planning, a filesystem for diffs, and review subagents.

## Consequences

Two prompts, two code paths, and two sets of tests to maintain, and `deepagents`
becomes a required dependency rather than an optional extra. We accepted this
over routing everything through the deep agent, which would make trivial
questions slow and expensive.
