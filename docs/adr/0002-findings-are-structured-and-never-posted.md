# Findings are structured data, and the Review Sheep library never writes to GitHub

A Review produces Findings as structured data (location, severity, description)
and a Report is rendered from them, so that findings can be filtered, asserted
on in tests, and later fed to other sinks. Review Sheep only ever reads from
GitHub: it does not submit PR reviews or inline comments, and needs no write
scope on its token.

## Consequences

Posting to GitHub stays outside the library, so a caller who wants it composes
it themselves from Findings. The bundled GitHub Actions workflow is such a
caller: after a successful CI Review, it upserts the rendered Report as one
pull-request conversation comment. Library and CLI consumers remain free of
implicit, author-notifying side effects.
