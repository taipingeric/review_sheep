"""Structured domain models returned by Review Sheep."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Severity(str, Enum):
    """How much a Finding matters, from blocking to cosmetic."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Confidence(str, Enum):
    """How sure a reviewer is that a Finding is real."""

    CONFIRMED = "confirmed"
    LIKELY = "likely"
    SPECULATIVE = "speculative"


class Lens(str, Enum):
    """The review perspective a Finding was produced through."""

    CORRECTNESS = "correctness"
    SECURITY = "security"
    CONVENTIONS_AND_TESTS = "conventions-and-tests"


class ReviewOperation(str, Enum):
    """The Review pipeline stage a ReviewError came from."""

    FETCH_SNAPSHOT = "fetch_snapshot"
    PREPARE_WORKSPACE = "prepare_workspace"
    RUN_REVIEW = "run_review"


class InquiryAnswer(BaseModel):
    """The answer to an Inquiry, or stable error data when it cannot run."""

    model_config = ConfigDict(frozen=True)

    text: str | None = None
    error: str | None = None
    incomplete: bool = False

    @model_validator(mode="after")
    def require_text_or_error(self) -> InquiryAnswer:
        """Require exactly one successful answer or failure description."""
        if (self.text is None) == (self.error is None):
            raise ValueError("InquiryAnswer requires exactly one of text or error")
        return self


class Location(BaseModel):
    """Where a Finding applies: whole snapshot, one file, or one line range."""

    model_config = ConfigDict(frozen=True)

    path: str | None = Field(default=None, min_length=1)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_granularity(self) -> Location:
        """Reject partial or inverted line ranges, and ranges without a path."""
        has_start = self.start_line is not None
        has_end = self.end_line is not None
        if has_start != has_end:
            raise ValueError("Location requires both start_line and end_line")
        if has_start:
            assert self.start_line is not None
            assert self.end_line is not None
            if self.path is None:
                raise ValueError("line-range Location requires a path")
            if self.end_line < self.start_line:
                raise ValueError("Location end_line must not precede start_line")
        return self


class Finding(BaseModel):
    """One actionable defect reported by a Review."""

    model_config = ConfigDict(frozen=True)

    description: str
    location: Location
    severity: Severity
    confidence: Confidence
    lens: Lens


class SnapshotFile(BaseModel):
    """One changed file as fetched from GitHub, including its raw patch."""

    model_config = ConfigDict(frozen=True)

    path: str
    status: str
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    patch: str
    previous_path: str | None = None


class PullRequestSnapshot(BaseModel):
    """One pull request's changed code pinned to a single head commit."""

    model_config = ConfigDict(frozen=True)

    repo: str
    number: int = Field(gt=0)
    head_sha: str
    files: list[SnapshotFile]


class ManifestFile(BaseModel):
    """One manifest entry pointing a reviewer at a virtual diff file."""

    model_config = ConfigDict(frozen=True)

    path: str
    status: str
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    changes: int = Field(ge=0)
    diff_path: str
    previous_path: str | None = None


class Manifest(BaseModel):
    """The index a reviewer reads first to navigate a snapshot."""

    model_config = ConfigDict(frozen=True)

    repo: str
    pull_request_number: int = Field(gt=0)
    head_sha: str
    files: list[ManifestFile]


class ReviewWorkspace(BaseModel):
    """A manifest plus the virtual files a Review Runner reads from."""

    model_config = ConfigDict(frozen=True)

    manifest: Manifest
    files: dict[str, str]

    def read(self, path: str) -> str:
        """Read one virtual Review file."""
        return self.files[path]


class Review(BaseModel):
    """A completed Review: what was reviewed and what was found."""

    model_config = ConfigDict(frozen=True)

    repo: str
    pull_request_number: int = Field(gt=0)
    head_sha: str
    manifest: Manifest
    findings: list[Finding]


class ReviewError(BaseModel):
    """Explainable failure data returned by the public Review boundary."""

    model_config = ConfigDict(frozen=True)

    repo: str
    pull_request_number: int = Field(gt=0)
    operation: ReviewOperation
    error_type: str
    message: str


class Report(BaseModel):
    """Human-facing presentation derived from a Review."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
