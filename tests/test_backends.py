from __future__ import annotations

from pathlib import Path

import pytest
from deepagents.backends.protocol import GlobResult

from review_sheep.backends import ReviewFilesystemBackend


def test_root_broad_glob_is_rejected_without_traversing(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass\n")
    backend = ReviewFilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    result = backend.glob("**/*.py")

    assert isinstance(result, GlobResult)
    assert result.matches == []
    assert result.truncated is True
    assert result.error is not None
    assert "root broad glob" in result.error


def test_scoped_globs_remain_available(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass\n")
    (tmp_path / "README.md").write_text("docs\n")
    backend = ReviewFilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    pattern_scoped = backend.glob("src/**/*.py")
    path_scoped = backend.glob("**/*.py", path="src")

    assert [match["path"] for match in pattern_scoped.matches or []] == ["/src/main.py"]
    assert [match["path"] for match in path_scoped.matches or []] == ["/src/main.py"]


def test_root_direct_file_glob_remains_available(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("pass\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "other.py").write_text("pass\n")
    backend = ReviewFilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    result = backend.glob("*.py")

    assert [match["path"] for match in result.matches or []] == ["/main.py"]


def test_exact_scoped_file_glob_remains_available(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass\n")
    backend = ReviewFilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    result = backend.glob("src/main.py")

    assert [match["path"] for match in result.matches or []] == ["/src/main.py"]


def test_glob_reports_missing_scope(tmp_path: Path) -> None:
    backend = ReviewFilesystemBackend(root_dir=tmp_path / "missing", virtual_mode=True)

    result = backend.glob("*.py")

    assert result.matches == []
    assert result.error is not None


@pytest.mark.parametrize("value", [0, -1, True])
def test_max_glob_results_requires_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ReviewFilesystemBackend(max_glob_results=value)  # type: ignore[arg-type]


def test_glob_results_are_capped_with_a_partial_result(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file-{index}.py").write_text("pass\n")
    backend = ReviewFilesystemBackend(
        root_dir=tmp_path,
        virtual_mode=True,
        max_glob_results=2,
    )

    result = backend.glob("*.py")

    assert len(result.matches or []) == 2
    assert result.truncated is True
    assert result.error is not None
    assert "too many results" in result.error
