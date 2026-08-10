# Diffs are fetched once onto the filesystem, with a manifest

Status: superseded by ADR-0007

The Review path fetches the pull request's diffs once, writes one file per
changed file into the deep agent's filesystem alongside a manifest listing the
changed paths and their line counts, and lens subagents read from there instead
of calling the GitHub API themselves. Every Lens therefore reviews the same
snapshot: were each Lens to fetch independently, a push mid-review would have
different lenses reviewing different commits and reporting Findings whose line
numbers disagree — an intermittent failure that is very hard to reproduce.

## Consequences

The Review path pays for a full fetch up front even when a Lens only cares about
a few files, in exchange for N rather than 3N API calls and a stable snapshot.
The manifest is what makes on-demand reading useful; without it a subagent has no
basis to skip a file.
