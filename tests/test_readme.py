from __future__ import annotations

import tomllib
from pathlib import Path


def test_readme_leads_install_with_the_published_package() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()

    assert "pip install review-sheep" in readme
    assert 'pip install "review-sheep[openai]"' in readme
    assert "To install from source instead:" in readme
    assert readme.index("pip install review-sheep") < readme.index("pip install .")


def test_readme_documents_explicit_secret_injection_and_env_example() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text()
    env_example = (root / ".env.example").read_text()

    assert "explicit configuration objects" in readme
    assert "never pass secrets as CLI" in readme
    assert "does not load a local `.env` file" in readme
    assert "REVIEW_LENS_CONCURRENCY=3" in env_example
    assert "OPENAI_API_KEY" not in env_example


def test_readme_documents_runtime_repository_cache_controls() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()

    assert "REVIEW_CACHE_ROOT" in readme
    assert "REVIEW_CACHE_CLEANUP" in readme
    assert "--no-cache" in readme
    assert "credential helper" in readme
    assert "cache logs report hits, misses" in readme.lower()


def test_project_documents_reproducible_coverage_checks() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())

    assert 'COVERAGE_FILE="${TMPDIR:-/tmp}/review-sheep.coverage"' in readme
    assert "uv run --extra dev --extra openai pytest" in readme
    assert "--cov=src/review_sheep" in readme
    assert "--cov-fail-under=97" in readme
    assert "97%" in readme
    assert "outside the source tree" in readme
    assert "statement coverage" in readme
    assert any(
        dependency.startswith("pytest-cov")
        for dependency in pyproject["project"]["optional-dependencies"]["dev"]
    )
    assert pyproject["tool"]["coverage"]["report"]["fail_under"] == 97


def test_coverage_workflow_enforces_the_floor_and_uses_runner_temp() -> None:
    workflow = (Path(__file__).parents[1] / ".github/workflows/test.yml").read_text()

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "COVERAGE_FILE: ${{ runner.temp }}/review-sheep.coverage" in workflow
    assert "--cov-report=term-missing" in workflow
    assert "--cov-fail-under=97" in workflow
