# Review subagents split by concern, not by file

Each review subagent gets one lens (correctness, security, conventions & tests)
and sees the whole pull request, reading diffs from the deep agent's filesystem
on demand rather than receiving them in its prompt. Splitting per file would make
cross-file inconsistencies — a changed caller with an unchanged callee — provably
undiscoverable, since no single subagent would ever see both sides.

## Consequences

Each lens carries the whole pull request, so very large pull requests stress a
single subagent's context. The architecture keeps room to subdivide within a lens
by file when that happens; the MVP does not.

The three Lenses run concurrently because they share no mutable state or data
dependency. Each worker receives a copied tracing context; after completion,
Findings are flattened in stable Lens order to keep Reports deterministic.
