from __future__ import annotations

from typing import Any

import pytest

from review_sheep import merge as merge_module


class _SingletonMergeAgent:
    """Keep every Finding apart so no test reaches a real model."""

    def invoke(self, state: dict[str, Any], **_: Any) -> dict[str, Any]:
        count = state["messages"][0]["content"].count('"index":')
        return {"structured_response": {"groups": [[i] for i in range(count)]}}


@pytest.fixture(autouse=True)
def _offline_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        merge_module, "_create_merge_agent", lambda model: _SingletonMergeAgent()
    )
