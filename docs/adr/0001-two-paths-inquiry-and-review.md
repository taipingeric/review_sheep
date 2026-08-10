# Two agent paths: Inquiry and Review

Answering "which PRs are open?" and reviewing a 40-file diff have very different
cost and capability needs, so we build two paths rather than one: an Inquiry path
on a plain tool-calling agent over read-only GitHub metadata, and a Review path
on a deep agent with planning, a filesystem for diffs, and review subagents.
`InquiryAgent` and the deep Review agents are LangChain agents. The conversational
entrypoint is a LangGraph whose Bot node delegates accumulated messages to
`InquiryAgent`. An IntentClassifier node routes changed-code requests to
ReviewBot and metadata requests to the Inquiry Bot. Review uses a separate
LangGraph for fixed-checkout validation and Lens execution; `ReviewAgent.review`
hides that graph behind a single structured interface.

## Consequences

Two prompts, two code paths, and two sets of tests to maintain, and `deepagents`
becomes a required dependency rather than an optional extra. We accepted this
over routing everything through the deep agent, which would make trivial
questions slow and expensive.
