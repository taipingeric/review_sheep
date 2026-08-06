# Python 3.11 floor, and no anthropic or gemini extras

`deepagents` requires Python `>=3.11` and already depends on
`langchain-anthropic` and `langchain-google-genai`, so taking it as a required
dependency drops our Python 3.10 support and makes those two optional extras
dishonest — installing the core already installs them. They are removed, along
with the `all` extra that referenced them; only `openai` remains an extra,
because `langchain-openai` is genuinely not pulled in by `deepagents` and
`models.py` needs it.

## Consequences

`langchain-core` moves to `>=1.5.0` to match what `deepagents` resolves to.
Anyone on Python 3.10 cannot use Review Sheep at all, and a future reader
wondering where the Anthropic extra went should look at `deepagents`'s own
dependency list rather than adding one back.
