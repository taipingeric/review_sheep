# Python 3.11 floor, and no anthropic or gemini extras

`deepagents` requires Python `>=3.11` and already installs its provider stack, so
taking it as a required dependency drops our Python 3.10 support and makes
provider optional extras dishonest. Anthropic is now also a direct dependency
because the production CI adapter imports `ChatAnthropic`; only `openai` remains
an extra because the optional `review_sheep.chat` adapter needs it.

## Consequences

`langchain-core` moves to `>=1.5.0` to match what `deepagents` resolves to.
Anyone on Python 3.10 cannot use Review Sheep at all, and a future reader
wondering where the Anthropic extra went should look at `deepagents`'s own
dependency list rather than adding one back.
