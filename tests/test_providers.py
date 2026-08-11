from __future__ import annotations

from typing import Any

import pytest

from review_sheep.providers import anthropic_model, parse_anthropic_custom_headers


def test_anthropic_custom_headers_use_claude_code_format() -> None:
    assert parse_anthropic_custom_headers(
        "X-Team: review-sheep\nAuthorization: Custom value\n\n"
    ) == {
        "X-Team": "review-sheep",
        "Authorization": "Custom value",
    }


@pytest.mark.parametrize("value", ["missing-colon", ": value", "Name:"])
def test_anthropic_custom_headers_reject_malformed_lines(value: str) -> None:
    with pytest.raises(ValueError, match="Name: Value"):
        parse_anthropic_custom_headers(value)


def test_anthropic_model_uses_bearer_auth_without_an_api_key_header() -> None:
    model = anthropic_model(
        model="gateway-sonnet",
        auth_token="auth-token",
        base_url="https://gateway.example.com",
        custom_headers="X-Team: review-sheep",
    )
    client_params: dict[str, Any] = model._client_params  # type: ignore[attr-defined]

    assert client_params["auth_token"] == "auth-token"
    assert "api_key" not in client_params
    assert client_params["base_url"] == "https://gateway.example.com"
    assert client_params["default_headers"]["X-Team"] == "review-sheep"
