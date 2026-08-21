from __future__ import annotations

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
