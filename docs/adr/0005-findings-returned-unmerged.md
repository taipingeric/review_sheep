# A Review returns every Lens's Findings unmerged

Findings from all lenses are returned as-is, each tagged with the Lens that
produced it, with no deduplication or verification step. The same problem
surfacing from two lenses appears twice. Merging is a semantic judgement — the
same missing permission check reads as a vulnerability to one Lens and a logic
error to another, with different wording, line ranges, and severities — and doing
it in the MVP would hide raw model output behind a step we cannot yet evaluate.

## Consequences

Reports contain duplicates, and the cost of reconciling them falls on the reader.
In exchange, Review output is exactly what the lenses said, which is what we need
in order to judge later whether semantic merging and a verification pass are
worth their tokens. The `lens` field on every Finding is what makes both
additions possible without changing the Finding type.
