# Review reads a fixed local Git checkout

This decision remains authoritative for CI and fixed-checkout library callers.
ADR 0008 adds a Manifest adapter for interactive Chat without weakening these
checkout invariants.

Review receives a clean local worktree whose `HEAD` exactly matches the pull
request head SHA and whose object database contains the base SHA. Lenses derive
their changed-file index and diffs from `git diff base...head` and can read the
complete head tree directly. This replaces GitHub patch fetching and the
generated Manifest: it preserves one immutable source across Lenses while also
making unchanged callers, callees, configuration, and tests available.

## Consequences

The caller or CI must prepare the checkout before Review, including fetching
both commits, and the run fails when the SHA or clean-worktree invariant is not
met. In exchange, Review avoids truncated API patches and duplicate source
representations. The checkout is confined and write-denied, but callers must
still keep credentials and other readable secrets outside it.
