from review_sheep import (
    Confidence,
    Finding,
    Lens,
    Location,
    Manifest,
    Report,
    Review,
    Severity,
    render_report,
)


def _review(findings: list[Finding]) -> Review:
    manifest = Manifest(
        repo="acme/widgets",
        pull_request_number=42,
        head_sha="abc123",
        files=[],
    )
    return Review(
        repo="acme/widgets",
        pull_request_number=42,
        head_sha="abc123",
        manifest=manifest,
        findings=findings,
    )


def test_report_renders_every_location_granularity_in_review_order() -> None:
    duplicate_description = "Authorization behavior changed without coverage."
    review = _review(
        [
            Finding(
                description="The pull request changes an undocumented API contract.",
                location=Location(),
                severity=Severity.MEDIUM,
                confidence=Confidence.LIKELY,
                lens=Lens.CORRECTNESS,
            ),
            Finding(
                description=duplicate_description,
                location=Location(path="src/auth.py"),
                severity=Severity.HIGH,
                confidence=Confidence.CONFIRMED,
                lens=Lens.SECURITY,
            ),
            Finding(
                description=duplicate_description,
                location=Location(path="tests/test_auth.py", start_line=21, end_line=21),
                severity=Severity.MEDIUM,
                confidence=Confidence.CONFIRMED,
                lens=Lens.CONVENTIONS_AND_TESTS,
            ),
            Finding(
                description="The changed branch can return an invalid state.",
                location=Location(path="src/state.py", start_line=8, end_line=12),
                severity=Severity.HIGH,
                confidence=Confidence.SPECULATIVE,
                lens=Lens.CORRECTNESS,
            ),
        ]
    )
    original = review.model_dump()

    report = render_report(review)

    assert isinstance(report, Report)
    assert report.text == """# Review Report: acme/widgets#42

Head: `abc123`
Findings: 4

## 1. MEDIUM — The pull request changes an undocumented API contract.

- Location: pull request
- Confidence: likely
- Lens: correctness

## 2. HIGH — Authorization behavior changed without coverage.

- Location: `src/auth.py` (whole file)
- Confidence: confirmed
- Lens: security

## 3. MEDIUM — Authorization behavior changed without coverage.

- Location: `tests/test_auth.py:21`
- Confidence: confirmed
- Lens: conventions-and-tests

## 4. HIGH — The changed branch can return an invalid state.

- Location: `src/state.py:8-12`
- Confidence: speculative
- Lens: correctness
"""
    assert report.text.count(duplicate_description) == 2
    assert review.model_dump() == original


def test_empty_review_has_explicit_report_output() -> None:
    report = render_report(_review([]))

    assert report.text == """# Review Report: acme/widgets#42

Head: `abc123`
Findings: 0

No Findings.
"""
