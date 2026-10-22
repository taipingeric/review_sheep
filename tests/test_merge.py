from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from review_sheep import (
    Confidence,
    CustomLens,
    Evidence,
    Finding,
    Lens,
    Location,
    Severity,
    github_line_url,
    render_report,
)
from review_sheep import merge as merge_module
from review_sheep.domain import Review
from review_sheep.lenses import lens_result_schema, lens_system_prompt
from review_sheep.merge import merge_findings


def _finding(
    description: str,
    *,
    lens: Lens | str,
    severity: Severity = Severity.MEDIUM,
    path: str = "src/auth.py",
    line: int = 1,
) -> Finding:
    return Finding(
        description=description,
        location=Location(path=path, start_line=line, end_line=line),
        severity=severity,
        confidence=Confidence.LIKELY,
        lens=lens,
    )


def _evidence(path: str, line: int, note: str) -> Evidence:
    return Evidence(
        location=Location(path=path, start_line=line, end_line=line), note=note
    )


class FakeMergeAgent:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"state": state, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return {"structured_response": self.response}


def _use_agent(monkeypatch: pytest.MonkeyPatch, agent: FakeMergeAgent) -> None:
    monkeypatch.setattr(merge_module, "_create_merge_agent", lambda model: agent)


def test_finding_defaults_lenses_to_its_own_lens() -> None:
    finding = _finding("Missing check.", lens=Lens.SECURITY)

    assert finding.title is None
    assert finding.evidence == []
    assert finding.lenses == [Lens.SECURITY]


def test_finding_rejects_lenses_without_its_lens() -> None:
    with pytest.raises(ValidationError):
        Finding(
            description="Missing check.",
            location=Location(),
            severity=Severity.HIGH,
            confidence=Confidence.LIKELY,
            lens=Lens.SECURITY,
            lenses=[Lens.CORRECTNESS],
        )


@pytest.mark.parametrize(
    "lens", [*Lens, CustomLens(identifier="api-compatibility", instructions="APIs.")]
)
def test_lens_schema_offers_title_and_evidence_but_hides_lenses(
    lens: Lens | CustomLens,
) -> None:
    schema = lens_result_schema(lens).model_json_schema()
    finding_schema = next(
        definition
        for name, definition in schema["$defs"].items()
        if name.endswith("Finding")
    )

    assert {"title", "evidence"} <= set(finding_schema["properties"])
    assert "suggestion" not in finding_schema["properties"]
    assert "lenses" not in finding_schema["properties"]
    assert "at most three sentences" in lens_system_prompt(lens)
    assert "evidence" in lens_system_prompt(lens)


def test_lens_result_fills_lenses_from_its_fixed_lens() -> None:
    result = lens_result_schema(Lens.CORRECTNESS).model_validate(
        {
            "findings": [
                {
                    "title": "Null caller crashes.",
                    "description": "The caller passes None.",
                    "evidence": [
                        {
                            "location": {
                                "path": "src/caller.py",
                                "start_line": 3,
                                "end_line": 3,
                            },
                            "note": "Passes None.",
                        }
                    ],
                    "location": {},
                    "severity": "high",
                    "confidence": "confirmed",
                }
            ]
        }
    )

    assert result.findings[0].lenses == [Lens.CORRECTNESS]
    assert result.findings[0].evidence[0].location.path == "src/caller.py"


@pytest.mark.parametrize("location", [Location(), Location(path="src/caller.py")])
def test_evidence_requires_a_linkable_line_range(location: Location) -> None:
    with pytest.raises(ValidationError):
        Evidence(location=location, note="Passes None.")


def test_github_line_url_links_a_single_line_or_a_range() -> None:
    def url(location: Location) -> str:
        return github_line_url(repo="acme/widgets", head_sha="abc", location=location)

    base = "https://github.com/acme/widgets/blob/abc"
    assert url(Location(path="src/a b.py", start_line=4, end_line=4)) == (
        f"{base}/src/a%20b.py#L4"
    )
    assert url(Location(path="src/a.py", start_line=4, end_line=9)) == (
        f"{base}/src/a.py#L4-L9"
    )
    assert url(Location(path="src/a.py")) == f"{base}/src/a.py"


def test_merge_keeps_highest_severity_and_records_every_lens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    findings = [
        _finding("Logic skips the check.", lens=Lens.CORRECTNESS, line=10),
        _finding("Unrelated typo.", lens=Lens.CORRECTNESS, severity=Severity.LOW),
        _finding(
            "Missing authorization.",
            lens=Lens.SECURITY,
            severity=Severity.CRITICAL,
            line=12,
        ),
        _finding("No test for the check.", lens=Lens.CONVENTIONS_AND_TESTS, line=40),
    ]
    agent = FakeMergeAgent({"groups": [[0, 2, 3], [1]]})
    _use_agent(monkeypatch, agent)

    merged = merge_findings(findings, model="openai:gpt-test")

    assert merged == [
        findings[2].model_copy(update={"lenses": list(Lens)}),
        findings[1],
    ]
    assert merged[0].lens is Lens.SECURITY
    request = agent.calls[0]["state"]["messages"][0]["content"]
    assert '"index": 3' in request
    assert agent.calls[0]["config"]["run_name"].endswith(":merge")


def test_merge_ignores_bad_indices_and_keeps_missing_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    findings = [
        _finding("A.", lens=Lens.CORRECTNESS),
        _finding("B.", lens=Lens.SECURITY),
        _finding("C.", lens=Lens.CONVENTIONS_AND_TESTS),
    ]
    _use_agent(monkeypatch, FakeMergeAgent({"groups": [[2, 7, -1], [2, 1]]}))

    merged = merge_findings(findings, model="openai:gpt-test")

    assert merged == findings


def test_merge_ties_keep_the_first_finding_text_and_pool_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _evidence("src/auth.py", 40, "Uses the value unchecked.")
    caller = _evidence("src/caller.py", 12, "Passes None.")
    findings = [
        _finding("First wording.", lens=Lens.CORRECTNESS).model_copy(
            update={"evidence": [shared]}
        ),
        _finding("Second wording.", lens="api-compatibility").model_copy(
            update={"evidence": [caller, shared]}
        ),
    ]
    _use_agent(monkeypatch, FakeMergeAgent({"groups": [[1, 0]]}))

    [merged] = merge_findings(findings, model="openai:gpt-test")

    assert merged.description == "First wording."
    assert merged.lenses == [Lens.CORRECTNESS, "api-compatibility"]
    assert merged.evidence == [shared, caller]


def test_merge_failure_returns_findings_unmerged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    findings = [
        _finding("A.", lens=Lens.CORRECTNESS),
        _finding("B.", lens=Lens.SECURITY),
    ]
    _use_agent(monkeypatch, FakeMergeAgent(RuntimeError("model down")))

    assert merge_findings(findings, model="openai:gpt-test") == findings


def test_merge_skips_the_model_for_fewer_than_two_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeMergeAgent({"groups": []})
    _use_agent(monkeypatch, agent)
    finding = _finding("A.", lens=Lens.CORRECTNESS)

    assert merge_findings([finding], model="openai:gpt-test") == [finding]
    assert agent.calls == []


def test_report_renders_title_linked_evidence_and_merged_lenses() -> None:
    finding = Finding(
        title="Delete skips authorization.",
        description="The delete handler never checks the owner.",
        evidence=[
            Evidence(
                location=Location(path="src/auth.py", start_line=12, end_line=14),
                note="Deletes without require_owner.",
            )
        ],
        location=Location(path="src/auth.py", start_line=12, end_line=12),
        severity=Severity.CRITICAL,
        confidence=Confidence.CONFIRMED,
        lens=Lens.SECURITY,
        lenses=[Lens.CORRECTNESS, Lens.SECURITY],
    )
    review = Review(
        repo="acme/widgets",
        pull_request_number=42,
        base_sha="base",
        head_sha="head",
        findings=[finding],
    )

    text = render_report(review).text

    assert "## 1. CRITICAL — Delete skips authorization.\n\n" in text
    assert "The delete handler never checks the owner.\n" in text
    assert "- Lenses: correctness, security\n" in text
    assert (
        "- Evidence:\n  - [`src/auth.py:12-14`]"
        "(https://github.com/acme/widgets/blob/head/src/auth.py#L12-L14)"
        " — Deletes without require_owner.\n"
    ) in text
    assert "Suggestion" not in text
