from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from review_sheep import (
    AcceptanceCriterion,
    CriterionVerdict,
    Decision,
    DecisionStatus,
    Finding,
    Review,
    ReviewBrief,
    ReviewCheckout,
    create_deep_review_agent,
    create_review_agent,
)
from review_sheep import review as review_module

BRIEF = {
    "intent": "Add a retry to the payout job. Does not change the schema.",
    "acceptance_criteria": [
        {
            "criterion": "A failed payout is retried at most three times.",
            "source": "pull_request",
            "quote": "Retry a failed payout up to three times.",
            "verdict": "met",
            "evidence": [
                {
                    "location": {"path": "job.py", "start_line": 4, "end_line": 9},
                    "note": "The loop stops at three attempts.",
                }
            ],
            "note": "The attempt counter is capped.",
        },
        {
            "criterion": "A retry is idempotent.",
            "source": "jira",
            "quote": "Retries must be idempotent.",
            "verdict": "unverified",
        },
    ],
    "decisions": [
        {
            "decision": "Retry in the job, not in the client.",
            "status": "needs_decision",
        }
    ],
}


def _checkout(root: Path) -> ReviewCheckout:
    return ReviewCheckout(
        repo="acme/widgets",
        pull_request_number=42,
        base_sha="base123",
        head_sha="head456",
        root=root,
        title="Retry payouts",
        body="Retry a failed payout up to three times.",
    )


class Source:
    def __init__(self, checkout: ReviewCheckout) -> None:
        self.checkout = checkout

    def prepare_checkout(self, *, repo: str, number: int) -> ReviewCheckout:
        return self.checkout


class PlainRunner:
    def run(self, checkout: ReviewCheckout) -> list[Finding]:
        return []


class BriefRunner(PlainRunner):
    brief_enabled = True

    def brief(self, checkout: ReviewCheckout) -> ReviewBrief:
        return ReviewBrief.model_validate(BRIEF)


class FailingBriefRunner(BriefRunner):
    def brief(self, checkout: ReviewCheckout) -> ReviewBrief:
        raise TimeoutError("model timed out")


class DisabledBriefRunner(BriefRunner):
    brief_enabled = False

    def brief(self, checkout: ReviewCheckout) -> ReviewBrief:
        raise AssertionError("brief must not run when disabled")


def test_review_without_a_brief_keeps_old_shape() -> None:
    review = Review(
        repo="acme/widgets",
        pull_request_number=1,
        base_sha="a",
        head_sha="b",
        findings=[],
    )

    assert review.intent == ""
    assert review.acceptance_criteria == []
    assert review.decisions == []


def test_criterion_rejects_an_unknown_verdict() -> None:
    with pytest.raises(ValidationError):
        AcceptanceCriterion(
            criterion="x",
            source="jira",
            quote="x",
            verdict="maybe",  # type: ignore[arg-type]
        )


def test_brief_models_validate_nested_evidence() -> None:
    brief = ReviewBrief.model_validate(BRIEF)

    assert brief.acceptance_criteria[0].verdict is CriterionVerdict.MET
    assert brief.acceptance_criteria[0].evidence[0].location.start_line == 4
    assert brief.acceptance_criteria[1].evidence == []
    assert isinstance(brief.decisions[0], Decision)


def test_review_carries_the_brief_from_an_opted_in_runner(tmp_path: Path) -> None:
    (tmp_path / "job.py").write_text("x = 1\n" * 20)
    result = create_review_agent(
        source=Source(_checkout(tmp_path)), runner=BriefRunner()
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert result.intent.startswith("Add a retry")
    assert [c.verdict for c in result.acceptance_criteria] == [
        CriterionVerdict.MET,
        CriterionVerdict.UNVERIFIED,
    ]
    assert result.decisions[0].decision == "Retry in the job, not in the client."


def test_review_with_no_findings_still_carries_the_brief(tmp_path: Path) -> None:
    result = create_review_agent(
        source=Source(_checkout(tmp_path)), runner=BriefRunner()
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert result.findings == []
    assert result.decisions


def test_a_runner_without_brief_support_still_works(tmp_path: Path) -> None:
    result = create_review_agent(
        source=Source(_checkout(tmp_path)), runner=PlainRunner()
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert result.intent == ""


def test_a_disabled_brief_is_not_run(tmp_path: Path) -> None:
    result = create_review_agent(
        source=Source(_checkout(tmp_path)), runner=DisabledBriefRunner()
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert result.acceptance_criteria == []


def test_a_failed_brief_does_not_fail_the_review(tmp_path: Path) -> None:
    result = create_review_agent(
        source=Source(_checkout(tmp_path)), runner=FailingBriefRunner()
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert result.intent == ""
    assert result.decisions == []
    assert result.brief_error == "TimeoutError: model timed out"


def test_brief_error_is_empty_when_the_brief_works_or_is_off(tmp_path: Path) -> None:
    for runner in (BriefRunner(), DisabledBriefRunner(), PlainRunner()):
        result = create_review_agent(
            source=Source(_checkout(tmp_path)), runner=runner
        ).review(repo="acme/widgets", number=42)

        assert isinstance(result, Review)
        assert result.brief_error == ""


class FakeBriefAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(state)
        return {"structured_response": BRIEF}


def test_deep_runner_brief_uses_one_agent_with_the_pr_description(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: list[dict[str, Any]] = []
    agent = FakeBriefAgent()

    def fake_create_deep_agent(**kwargs: Any) -> FakeBriefAgent:
        created.append(kwargs)
        return agent

    monkeypatch.setattr(review_module, "_create_deep_agent", fake_create_deep_agent)

    brief = review_module.DeepAgentReviewRunner(
        model="openai:gpt-test", brief=True
    ).brief(_checkout(tmp_path))

    assert len(created) == 1
    assert "Review Brief" in created[0]["system_prompt"]
    assert created[0]["permissions"][0].mode == "deny"
    request = agent.calls[0]["messages"][0]["content"]
    assert "Retry a failed payout up to three times." in request
    assert brief.acceptance_criteria[0].verdict is CriterionVerdict.MET


def test_the_deep_review_agent_writes_a_brief_by_default_and_can_opt_out() -> None:
    on = create_deep_review_agent(source=Source.__new__(Source), model="m")
    off = create_deep_review_agent(
        source=Source.__new__(Source), model="m", brief=False
    )

    assert off is not None and on is not None
    assert review_module.DeepAgentReviewRunner(model="m").brief_enabled is True
    assert not review_module.DeepAgentReviewRunner(model="m", brief=False).brief_enabled


def _review(**extra: Any) -> Review:
    return Review(
        repo="acme/widgets",
        pull_request_number=42,
        base_sha="base123",
        head_sha="head456",
        findings=[],
        **extra,
    )


def test_report_renders_the_brief() -> None:
    from review_sheep.report import render_report

    brief = ReviewBrief.model_validate(BRIEF)
    text = render_report(_review(**brief.model_dump())).text

    assert "## Intent" in text
    assert "✅ met — A failed payout is retried at most three times." in text
    assert "❔ unverified — A retry is idempotent." in text
    assert "job.py:4-9" in text
    assert "- Retry in the job, not in the client." in text
    assert "Review Brief failed" not in text


def test_report_shows_a_failed_brief() -> None:
    from review_sheep.report import render_report

    text = render_report(_review(brief_error="TimeoutError: model timed out")).text

    assert "> ⚠️ Review Brief failed: TimeoutError: model timed out" in text


def test_report_without_a_brief_has_no_brief_sections() -> None:
    from review_sheep.report import render_report

    text = render_report(_review()).text

    assert "## Intent" not in text
    assert "## Acceptance Criteria" not in text
    assert "## Decisions" not in text


def _brief_citing(path: str, start: int, end: int, verdict: str = "met") -> ReviewBrief:
    cited = {"location": {"path": path, "start_line": start, "end_line": end}}
    cited["note"] = "shows it"
    return ReviewBrief.model_validate(
        {
            "acceptance_criteria": [
                {
                    "criterion": "c",
                    "source": "jira",
                    "quote": "q",
                    "verdict": verdict,
                    "evidence": [cited],
                }
            ],
            "decisions": [
                {"decision": "d", "status": "needs_decision", "evidence": [cited]}
            ],
        }
    )


def test_verify_keeps_evidence_that_exists(tmp_path: Path) -> None:
    from review_sheep.brief import verify_brief

    (tmp_path / "job.py").write_text("x = 1\n" * 10)

    brief = verify_brief(_brief_citing("job.py", 4, 9), tmp_path)

    assert brief.acceptance_criteria[0].verdict is CriterionVerdict.MET
    assert len(brief.acceptance_criteria[0].evidence) == 1
    assert len(brief.decisions[0].evidence) == 1


@pytest.mark.parametrize(
    ("path", "start", "end"),
    [
        ("job.py", 4, 999),  # past the end of the file
        ("missing.py", 1, 2),  # no such file
        ("../outside.py", 1, 1),  # outside the checkout
    ],
)
def test_verify_drops_evidence_that_does_not_exist(
    tmp_path: Path, path: str, start: int, end: int
) -> None:
    from review_sheep.brief import verify_brief

    root = tmp_path / "repo"
    root.mkdir()
    (root / "job.py").write_text("x = 1\n" * 10)
    (tmp_path / "outside.py").write_text("x = 1\n" * 10)

    brief = verify_brief(_brief_citing(path, start, end), root)

    criterion = brief.acceptance_criteria[0]
    assert criterion.evidence == []
    assert criterion.verdict is CriterionVerdict.UNVERIFIED
    assert "No cited line exists" in criterion.note
    assert brief.decisions[0].evidence == []


def test_verify_leaves_a_not_met_verdict_alone(tmp_path: Path) -> None:
    from review_sheep.brief import verify_brief

    brief = verify_brief(_brief_citing("missing.py", 1, 2, "not_met"), tmp_path)

    assert brief.acceptance_criteria[0].verdict is CriterionVerdict.NOT_MET


def test_lenses_and_brief_run_at_the_same_time(tmp_path: Path) -> None:
    import threading

    # Each side waits for the other; this only passes if both run in parallel.
    meeting = threading.Barrier(2, timeout=5)

    class ParallelRunner(BriefRunner):
        def run(self, checkout: ReviewCheckout) -> list[Finding]:
            meeting.wait()
            return []

        def brief(self, checkout: ReviewCheckout) -> ReviewBrief:
            meeting.wait()
            return ReviewBrief()

    result = create_review_agent(
        source=Source(_checkout(tmp_path)), runner=ParallelRunner()
    ).review(repo="acme/widgets", number=42)

    assert isinstance(result, Review)
    assert result.brief_error == ""


def _brief_quoting(*quotes: str) -> ReviewBrief:
    return ReviewBrief.model_validate(
        {
            "acceptance_criteria": [
                {
                    "criterion": f"c{index}",
                    "source": "jira",
                    "quote": quote,
                    "verdict": "met",
                }
                for index, quote in enumerate(quotes)
            ]
        }
    )


def test_stated_criteria_keep_a_quote_found_in_the_named_source() -> None:
    from review_sheep.brief import keep_stated_criteria

    brief = keep_stated_criteria(
        _brief_quoting("must  RETRY thrice"),
        {"jira": "ABC-1\nThe job must retry\nthrice on failure."},
    )

    assert [c.criterion for c in brief.acceptance_criteria] == ["c0"]


def test_stated_criteria_drop_a_quote_the_source_never_said() -> None:
    from review_sheep.brief import keep_stated_criteria

    brief = keep_stated_criteria(
        _brief_quoting("must retry thrice", "must log every retry"),
        {"jira": "The job must retry thrice on failure."},
    )

    assert [c.criterion for c in brief.acceptance_criteria] == ["c0"]


def test_stated_criteria_drop_a_quote_filed_under_the_wrong_source() -> None:
    from review_sheep.brief import keep_stated_criteria

    # The words are real, but they are in the PR text, not in Jira.
    brief = keep_stated_criteria(
        _brief_quoting("must retry thrice"),
        {"jira": "Something else.", "pull_request": "It must retry thrice."},
    )

    assert brief.acceptance_criteria == []


def test_stated_criteria_drop_an_unknown_source() -> None:
    from review_sheep.brief import keep_stated_criteria

    brief = keep_stated_criteria(
        _brief_quoting("must retry thrice"), {"notion": "It must retry thrice."}
    )

    assert brief.acceptance_criteria == []


def test_report_says_when_no_acceptance_criteria_were_found() -> None:
    from review_sheep.report import render_report

    text = render_report(_review(intent="Add a retry.")).text

    assert "No acceptance criteria found in the provided documents." in text


def test_report_does_not_claim_missing_criteria_without_a_brief() -> None:
    from review_sheep.report import render_report

    plain = render_report(_review()).text
    failed = render_report(_review(brief_error="TimeoutError: x")).text

    assert "No acceptance criteria found" not in plain
    assert "No acceptance criteria found" not in failed


def test_report_names_where_a_criterion_is_written() -> None:
    from review_sheep.report import render_report

    brief = ReviewBrief.model_validate(BRIEF)
    text = render_report(_review(**brief.model_dump())).text

    assert "retried at most three times. (pull_request)" in text
    assert "A retry is idempotent. (jira)" in text


def _decision_brief(source: str, quote: str) -> ReviewBrief:
    return ReviewBrief.model_validate(
        {
            "decisions": [
                {
                    "decision": "Retries run in the job.",
                    "status": "confirmed",
                    "source": source,
                    "quote": quote,
                }
            ]
        }
    )


def test_a_decision_needs_a_status() -> None:
    with pytest.raises(ValueError):
        ReviewBrief.model_validate({"decisions": [{"decision": "d"}]})


def test_a_confirmed_decision_needs_a_source_and_quote() -> None:
    with pytest.raises(ValueError):
        ReviewBrief.model_validate(
            {"decisions": [{"decision": "d", "status": "confirmed"}]}
        )


def test_a_needs_decision_decision_needs_no_source() -> None:
    brief = ReviewBrief.model_validate(
        {"decisions": [{"decision": "d", "status": "needs_decision"}]}
    )

    assert brief.decisions[0].status is DecisionStatus.NEEDS_DECISION


def test_a_confirmed_decision_stays_when_the_file_has_the_quote(
    tmp_path: Path,
) -> None:
    from review_sheep.brief import check_decisions

    (tmp_path / "docs" / "adr").mkdir(parents=True)
    (tmp_path / "docs/adr/0001.md").write_text("We retry in the\njob, not the client.")

    brief = check_decisions(
        _decision_brief("docs/adr/0001.md", "retry in the job"), {}, tmp_path
    )

    assert brief.decisions[0].status is DecisionStatus.CONFIRMED
    assert brief.decisions[0].source == "docs/adr/0001.md"


def test_a_confirmed_decision_stays_when_a_document_has_the_quote(
    tmp_path: Path,
) -> None:
    from review_sheep.brief import check_decisions

    brief = check_decisions(
        _decision_brief("pull_request", "Retry in the job"),
        {"pull_request": "Title\nRetry in the job, not the client."},
        tmp_path,
    )

    assert brief.decisions[0].status is DecisionStatus.CONFIRMED


@pytest.mark.parametrize(
    ("source", "quote"),
    [
        ("docs/adr/0001.md", "a made-up sentence"),  # quote not in the file
        ("docs/adr/missing.md", "retry in the job"),  # no such file
        ("../outside.md", "retry in the job"),  # outside the checkout
        ("pull_request", "a made-up sentence"),  # quote not in the document
    ],
)
def test_an_unverifiable_confirmed_decision_becomes_needs_decision(
    tmp_path: Path, source: str, quote: str
) -> None:
    from review_sheep.brief import check_decisions

    root = tmp_path / "repo"
    (root / "docs" / "adr").mkdir(parents=True)
    (root / "docs/adr/0001.md").write_text("We retry in the job, not the client.")
    (tmp_path / "outside.md").write_text("We retry in the job, not the client.")

    brief = check_decisions(
        _decision_brief(source, quote), {"pull_request": "Something else."}, root
    )

    decision = brief.decisions[0]
    assert decision.status is DecisionStatus.NEEDS_DECISION
    assert decision.decision == "Retries run in the job."
    assert decision.source == ""
    assert decision.quote == ""


def test_report_groups_decisions_by_status() -> None:
    from review_sheep.report import render_report

    review = _review(
        decisions=[
            {"decision": "Ask first.", "status": "needs_decision"},
            {
                "decision": "Retry in the job.",
                "status": "confirmed",
                "source": "docs/adr/0001.md",
                "quote": "We retry\nin the job.",
            },
        ]
    )

    text = render_report(review).text

    assert "## Confirmed decisions" in text
    assert "## User decision needed" in text
    assert text.index("## Confirmed decisions") < text.index("## User decision needed")
    assert "- Retry in the job. (docs/adr/0001.md)" in text
    assert "> We retry in the job." in text
    assert "- Ask first." in text
    assert "## Decisions" not in text


def test_report_omits_an_empty_decision_group() -> None:
    from review_sheep.report import render_report

    text = render_report(
        _review(decisions=[{"decision": "d", "status": "needs_decision"}])
    ).text

    assert "## User decision needed" in text
    assert "## Confirmed decisions" not in text
