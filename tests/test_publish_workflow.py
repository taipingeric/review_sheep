from __future__ import annotations

from pathlib import Path


def test_publish_workflow_builds_and_publishes_via_trusted_publishing() -> None:
    workflow = (Path(__file__).parents[1] / ".github/workflows/publish.yml").read_text()

    assert "release:" in workflow
    assert "types: [published]" in workflow
    assert "tags:" in workflow
    assert "'v*'" in workflow or '"v*"' in workflow

    assert "contents: read" in workflow
    assert "id-token: write" in workflow
    assert "uv build" in workflow
    assert "pypa/gh-action-pypi-publish" in workflow
    assert "skip-existing: true" in workflow

    assert "secrets.PYPI" not in workflow
    assert "password:" not in workflow

    # Job-level `permissions:` replaces workflow-level ones for that job in
    # GitHub Actions, so a workflow-level block here would be silently dead.
    assert "\npermissions:\n" not in workflow
