from __future__ import annotations

import pytest
from pydantic import ValidationError

from review_sheep import (
    Confidence,
    CustomLens,
    Finding,
    Lens,
    Location,
    Severity,
    validate_custom_lenses,
)


def test_custom_lens_is_immutable_and_preserves_its_public_contract() -> None:
    custom_lens = CustomLens(
        identifier="api-compatibility",
        instructions="Review public API changes for compatibility regressions.",
    )

    assert custom_lens.identifier == "api-compatibility"
    assert custom_lens.instructions == (
        "Review public API changes for compatibility regressions."
    )
    assert custom_lens.model_dump() == {
        "identifier": "api-compatibility",
        "instructions": "Review public API changes for compatibility regressions.",
    }

    with pytest.raises(ValidationError):
        custom_lens.identifier = "different-lens"


@pytest.mark.parametrize(
    "identifier",
    ["", " ", "API-compatibility", "api_compatibility", "api compatibility"],
)
def test_custom_lens_rejects_invalid_identifiers(identifier: str) -> None:
    with pytest.raises(ValidationError):
        CustomLens(identifier=identifier, instructions="Review API compatibility.")


@pytest.mark.parametrize("identifier", [lens.value for lens in Lens])
def test_custom_lens_rejects_built_in_identifiers(identifier: str) -> None:
    with pytest.raises(ValidationError, match="built-in"):
        CustomLens(identifier=identifier, instructions="Review this concern.")


def test_custom_lens_rejects_blank_instructions() -> None:
    with pytest.raises(ValidationError):
        CustomLens(identifier="api-compatibility", instructions=" ")


def test_custom_lens_collection_rejects_duplicate_identifiers() -> None:
    lenses = [
        CustomLens(
            identifier="api-compatibility",
            instructions="Review API compatibility.",
        ),
        CustomLens(
            identifier="api-compatibility",
            instructions="Review API compatibility a second time.",
        ),
    ]

    with pytest.raises(ValueError, match="duplicate"):
        validate_custom_lenses(lenses)


def test_custom_lens_collection_preserves_declaration_order() -> None:
    lenses = [
        CustomLens(identifier="api-compatibility", instructions="Review APIs."),
        CustomLens(identifier="migration-safety", instructions="Review migrations."),
    ]

    assert [lens.identifier for lens in validate_custom_lenses(lenses)] == [
        "api-compatibility",
        "migration-safety",
    ]


def test_finding_preserves_built_in_and_custom_lens_identities() -> None:
    built_in = Finding(
        description="A built-in Finding.",
        location=Location(),
        severity=Severity.LOW,
        confidence=Confidence.CONFIRMED,
        lens=Lens.CORRECTNESS,
    )
    custom = Finding(
        description="A custom Finding.",
        location=Location(),
        severity=Severity.MEDIUM,
        confidence=Confidence.LIKELY,
        lens="api-compatibility",
    )

    assert built_in.lens is Lens.CORRECTNESS
    assert custom.lens == "api-compatibility"
    assert built_in.model_dump(mode="json")["lens"] == "correctness"
    assert custom.model_dump(mode="json")["lens"] == "api-compatibility"


def test_finding_converts_built_in_string_to_its_lens_identity() -> None:
    finding = Finding(
        description="A built-in Finding.",
        location=Location(),
        severity=Severity.LOW,
        confidence=Confidence.CONFIRMED,
        lens="correctness",
    )

    assert finding.lens is Lens.CORRECTNESS


@pytest.mark.parametrize("lens", ["", "API-compatibility", "api_compatibility"])
def test_finding_rejects_invalid_custom_lens_identity(lens: str) -> None:
    with pytest.raises(ValidationError):
        Finding(
            description="An invalidly identified Finding.",
            location=Location(),
            severity=Severity.LOW,
            confidence=Confidence.CONFIRMED,
            lens=lens,
        )
