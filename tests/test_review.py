from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from review_sheep import (
    Confidence,
    DeepAgentReviewRunner,
    Finding,
    Lens,
    Location,
    PullRequestSnapshot,
    Review,
    ReviewError,
    ReviewOperation,
    ReviewWorkspace,
    Severity,
    SnapshotFile,
    create_deep_review_agent,
    create_review_agent,
)
from review_sheep import review as review_module


class FakeSnapshotSource:
    def __init__(self, snapshot: PullRequestSnapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[dict[str, Any]] = []

    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        self.calls.append({"repo": repo, "number": number})
        return self.snapshot


class DeterministicReviewRunner:
    def __init__(self) -> None:
        self.workspaces: list[ReviewWorkspace] = []

    def run(self, workspace: ReviewWorkspace) -> list[Finding]:
        self.workspaces.append(workspace)
        caller = workspace.read("/diffs/src/caller.py.diff")
        callee = workspace.read("/diffs/src/callee.py.diff")
        assert "normalize_user(user)" in caller
        assert "def normalize_user(user, context)" in callee
        return [
            Finding(
                description=(
                    "normalize_user now requires context, but its caller still passes "
                    "only user."
                ),
                location=Location(path="src/caller.py", start_line=12, end_line=12),
                severity=Severity.HIGH,
                confidence=Confidence.CONFIRMED,
                lens=Lens.CORRECTNESS,
            )
        ]


class FakeDeepAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(state)
        return {
            "structured_response": {
                "findings": [
                    {
                        "description": "The changed caller omits the required context.",
                        "location": {
                            "path": "src/caller.py",
                            "start_line": 12,
                            "end_line": 12,
                        },
                        "severity": "high",
                        "confidence": "confirmed",
                        "lens": "correctness",
                    }
                ]
            }
        }


class FailingSnapshotSource:
    def fetch_snapshot(self, *, repo: str, number: int) -> PullRequestSnapshot:
        raise RuntimeError("GitHub is unavailable")


class FailingReviewRunner:
    def run(self, workspace: ReviewWorkspace) -> list[Finding]:
        raise TimeoutError("model timed out")


def test_review_returns_a_cross_file_correctness_finding_from_one_snapshot() -> None:
    snapshot = PullRequestSnapshot(
        repo="acme/widgets",
        number=42,
        head_sha="abc123",
        files=[
            SnapshotFile(
                path="src/caller.py",
                status="modified",
                additions=1,
                deletions=1,
                patch="@@ -12 +12 @@\n-normalize_user(user, context)\n+normalize_user(user)",
            ),
            SnapshotFile(
                path="src/callee.py",
                status="modified",
                additions=1,
                deletions=1,
                patch=(
                    "@@ -3 +3 @@\n-def normalize_user(user):\n"
                    "+def normalize_user(user, context):"
                ),
            ),
        ],
    )
    source = FakeSnapshotSource(snapshot)
    runner = DeterministicReviewRunner()

    review = create_review_agent(source=source, runner=runner).review(
        repo="acme/widgets", number=42
    )

    assert isinstance(review, Review)
    assert source.calls == [{"repo": "acme/widgets", "number": 42}]
    assert len(runner.workspaces) == 1
    assert runner.workspaces[0].manifest.model_dump() == {
        "repo": "acme/widgets",
        "pull_request_number": 42,
        "head_sha": "abc123",
        "files": [
            {
                "path": "src/caller.py",
                "status": "modified",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
                "diff_path": "/diffs/src/caller.py.diff",
                "previous_path": None,
            },
            {
                "path": "src/callee.py",
                "status": "modified",
                "additions": 1,
                "deletions": 1,
                "changes": 2,
                "diff_path": "/diffs/src/callee.py.diff",
                "previous_path": None,
            },
        ],
    }
    assert review.repo == "acme/widgets"
    assert review.pull_request_number == 42
    assert review.head_sha == "abc123"
    assert review.manifest == runner.workspaces[0].manifest
    assert review.findings == [
        Finding(
            description=(
                "normalize_user now requires context, but its caller still passes only user."
            ),
            location=Location(path="src/caller.py", start_line=12, end_line=12),
            severity=Severity.HIGH,
            confidence=Confidence.CONFIRMED,
            lens=Lens.CORRECTNESS,
        )
    ]


@pytest.mark.parametrize(
    "values",
    [
        {"start_line": 1, "end_line": 1},
        {"path": "src/caller.py", "start_line": 1},
        {"path": "src/caller.py", "start_line": 0, "end_line": 1},
        {"path": "src/caller.py", "start_line": 12, "end_line": 11},
    ],
)
def test_location_rejects_incomplete_or_invalid_line_ranges(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        Location(**values)


def test_deep_agent_runner_receives_the_workspace_and_returns_findings() -> None:
    snapshot = PullRequestSnapshot(
        repo="acme/widgets",
        number=42,
        head_sha="abc123",
        files=[
            SnapshotFile(
                path="src/caller.py",
                status="modified",
                additions=1,
                deletions=1,
                patch="@@ -12 +12 @@\n-old\n+new",
            )
        ],
    )
    source = FakeSnapshotSource(snapshot)
    deep_agent = FakeDeepAgent()
    runner = DeepAgentReviewRunner(agent=deep_agent)

    review = create_review_agent(source=source, runner=runner).review(
        repo="acme/widgets", number=42
    )

    assert isinstance(review, Review)
    assert len(deep_agent.calls) == 1
    files = deep_agent.calls[0]["files"]
    assert set(files) == {"/manifest.json", "/diffs/src/caller.py.diff"}
    assert "abc123" in str(files["/manifest.json"])
    assert "@@ -12 +12 @@" in str(files["/diffs/src/caller.py.diff"])
    assert review.findings == [
        Finding(
            description="The changed caller omits the required context.",
            location=Location(path="src/caller.py", start_line=12, end_line=12),
            severity=Severity.HIGH,
            confidence=Confidence.CONFIRMED,
            lens=Lens.CORRECTNESS,
        )
    ]


def test_deep_review_agent_delegates_every_lens_with_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = PullRequestSnapshot(
        repo="acme/widgets",
        number=42,
        head_sha="abc123",
        files=[
            SnapshotFile(
                path="src/caller.py",
                status="modified",
                additions=1,
                deletions=1,
                patch="@@ -1 +1 @@\n-old\n+new",
            )
        ],
    )
    source = FakeSnapshotSource(snapshot)
    deep_agent = FakeDeepAgent()
    factory_calls: list[dict[str, Any]] = []

    def fake_create_deep_agent(**kwargs: Any) -> FakeDeepAgent:
        factory_calls.append(kwargs)
        return deep_agent

    monkeypatch.setattr(review_module, "_create_deep_agent", fake_create_deep_agent)

    review = create_deep_review_agent(source=source, model="openai:gpt-5-mini").review(
        repo="acme/widgets", number=42
    )

    assert isinstance(review, Review)
    assert len(factory_calls) == 1
    call = factory_calls[0]
    assert call["model"] == "openai:gpt-5-mini"
    assert call["backend"].__class__.__name__ == "StateBackend"
    assert call["response_format"] is review_module.ReviewFindings
    assert call.get("tools") is None
    assert len(call["subagents"]) == 3
    assert [subagent["name"] for subagent in call["subagents"]] == [
        "correctness-reviewer",
        "security-reviewer",
        "conventions-and-tests-reviewer",
    ]
    correctness = call["subagents"][0]
    assert correctness["response_format"] is review_module.CorrectnessResult
    assert "whole pull request" in correctness["system_prompt"]
    assert call["subagents"][1]["response_format"] is review_module.SecurityResult
    assert (
        call["subagents"][2]["response_format"]
        is review_module.ConventionsAndTestsResult
    )
    assert all("tools" not in subagent for subagent in call["subagents"])
    assert review.findings[0].lens is Lens.CORRECTNESS


def test_lens_output_schema_rejects_another_lens_identity() -> None:
    with pytest.raises(ValidationError):
        review_module.SecurityResult.model_validate(
            {
                "findings": [
                    {
                        "description": "Wrongly tagged security finding.",
                        "location": {},
                        "severity": "high",
                        "confidence": "confirmed",
                        "lens": "correctness",
                    }
                ]
            }
        )


def test_review_returns_github_failure_as_explainable_pydantic_data() -> None:
    result = create_review_agent(
        source=FailingSnapshotSource(),
        runner=DeterministicReviewRunner(),
    ).review(repo="acme/widgets", number=42)

    assert result == ReviewError(
        repo="acme/widgets",
        pull_request_number=42,
        operation=ReviewOperation.FETCH_SNAPSHOT,
        error_type="RuntimeError",
        message="GitHub is unavailable",
    )


def test_review_returns_model_failure_as_explainable_pydantic_data() -> None:
    source = FakeSnapshotSource(
        PullRequestSnapshot(
            repo="acme/widgets",
            number=42,
            head_sha="abc123",
            files=[],
        )
    )

    result = create_review_agent(
        source=source,
        runner=FailingReviewRunner(),
    ).review(repo="acme/widgets", number=42)

    assert result == ReviewError(
        repo="acme/widgets",
        pull_request_number=42,
        operation=ReviewOperation.RUN_REVIEW,
        error_type="TimeoutError",
        message="model timed out",
    )


class DeterministicAllLensRunner:
    def __init__(self) -> None:
        self.workspaces: list[ReviewWorkspace] = []

    def run(self, workspace: ReviewWorkspace) -> list[Finding]:
        self.workspaces.append(workspace)
        location = Location(path="src/auth.py", start_line=8, end_line=8)
        return [
            Finding(
                description="The caller no longer supplies the required actor.",
                location=location,
                severity=Severity.HIGH,
                confidence=Confidence.CONFIRMED,
                lens=Lens.CORRECTNESS,
            ),
            Finding(
                description="The missing actor bypasses the authorization decision.",
                location=location,
                severity=Severity.CRITICAL,
                confidence=Confidence.LIKELY,
                lens=Lens.SECURITY,
            ),
            Finding(
                description="The authorization regression has no failing test.",
                location=Location(path="tests/test_auth.py"),
                severity=Severity.MEDIUM,
                confidence=Confidence.CONFIRMED,
                lens=Lens.CONVENTIONS_AND_TESTS,
            ),
        ]


def test_review_preserves_findings_from_every_lens_without_merging() -> None:
    source = FakeSnapshotSource(
        PullRequestSnapshot(
            repo="acme/widgets",
            number=42,
            head_sha="abc123",
            files=[
                SnapshotFile(
                    path="src/auth.py",
                    status="modified",
                    additions=1,
                    deletions=1,
                    patch="@@ -8 +8 @@\n-authorize(actor)\n+authorize()",
                ),
                SnapshotFile(
                    path="tests/test_auth.py",
                    status="modified",
                    additions=1,
                    deletions=0,
                    patch="@@ -20,0 +21 @@\n+assert response.ok",
                ),
            ],
        )
    )
    runner = DeterministicAllLensRunner()

    result = create_review_agent(source=source, runner=runner).review(
        repo="acme/widgets", number=42
    )

    assert isinstance(result, Review)
    assert source.calls == [{"repo": "acme/widgets", "number": 42}]
    assert len(runner.workspaces) == 1
    assert [finding.lens for finding in result.findings] == [
        Lens.CORRECTNESS,
        Lens.SECURITY,
        Lens.CONVENTIONS_AND_TESTS,
    ]
    assert result.findings[0].location == result.findings[1].location
    assert len(result.findings) == 3
