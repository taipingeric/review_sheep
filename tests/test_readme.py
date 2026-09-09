from __future__ import annotations

import tomllib
from pathlib import Path


def test_readme_leads_install_with_the_published_package() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()

    assert "pip install review-sheep" in readme
    assert 'pip install "review-sheep[openai]"' in readme
    assert "To install from source instead:" in readme
    assert readme.index("pip install review-sheep") < readme.index("pip install .")


def test_readme_documents_explicit_secret_injection_without_dotenv() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text()

    assert "explicit configuration objects" in readme
    assert "never pass secrets as CLI" in readme
    assert "do not require a checked-out" in readme
    assert not (root / ".env.example").exists()


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

    assert (
        "uv run --extra dev pytest --cov=src/review_sheep --cov-report=term-missing"
        in readme
    )
    assert "baseline" in readme
    assert "104 tests" in readme
    assert "94%" in readme
    assert "statement coverage" in readme
    assert any(
        dependency.startswith("pytest-cov")
        for dependency in pyproject["project"]["optional-dependencies"]["dev"]
    )
