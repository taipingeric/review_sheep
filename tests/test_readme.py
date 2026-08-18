from __future__ import annotations

from pathlib import Path


def test_readme_leads_install_with_the_published_package() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()

    assert "pip install review-sheep" in readme
    assert 'pip install "review-sheep[openai]"' in readme
    assert "To install from source instead:" in readme
    assert readme.index("pip install review-sheep") < readme.index("pip install .")
