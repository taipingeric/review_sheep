# Findings are structured data, and Review Sheep never writes to GitHub

A Review produces Findings as structured data (location, severity, description)
and a Report is rendered from them, so that findings can be filtered, asserted
on in tests, and later fed to other sinks. Review Sheep only ever reads from
GitHub: it does not submit PR reviews or inline comments, and needs no write
scope on its token.

## Consequences

Posting to GitHub stays outside the library, so a caller who wants it composes
it themselves from Findings. This keeps every run of Review Sheep free of
irreversible, author-notifying side effects.
