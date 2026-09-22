"""Review-specific filesystem backend policies."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import GlobResult


class ReviewFilesystemBackend(FilesystemBackend):
    """Bound repository exploration without changing filesystem semantics."""

    MAX_GLOB_RESULTS = 100

    def __init__(
        self,
        root_dir: str | Path | None = None,
        virtual_mode: bool = True,
        max_file_size_mb: int = 10,
        *,
        max_glob_results: int = MAX_GLOB_RESULTS,
    ) -> None:
        if isinstance(max_glob_results, bool) or max_glob_results < 1:
            raise ValueError("max_glob_results must be a positive integer")
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            max_file_size_mb=max_file_size_mb,
        )
        self.max_glob_results = max_glob_results

    def glob(
        self,
        pattern: str,
        path: str | None = None,
    ) -> GlobResult:
        """Reject root-wide recursive searches and cap returned matches."""
        if _is_root_wide_glob(pattern, path):
            return GlobResult(
                matches=[],
                truncated=True,
                error=(
                    "Repository-root broad glob is disabled. Start with "
                    "list_changed_files, then search a specific directory "
                    "with a narrow pattern."
                ),
            )

        scoped_path, scoped_pattern = _narrow_glob_scope(pattern, path)
        if scoped_path is not None and scoped_pattern is not None:
            return self.glob(scoped_pattern, path=scoped_path)

        normalized_pattern = pattern.strip().replace("\\", "/").lstrip("/")
        if "/" not in normalized_pattern and normalized_pattern != "**":
            return self._limit_glob_results(
                self._glob_direct_children(normalized_pattern, path)
            )

        return self._limit_glob_results(super().glob(pattern, path))

    def _glob_direct_children(
        self,
        pattern: str,
        path: str | None,
    ) -> GlobResult:
        listing = self.ls(path or "/")
        if listing.entries is None:
            return GlobResult(error=listing.error, matches=[])

        matches = [
            entry
            for entry in listing.entries
            if not entry["is_dir"]
            and fnmatchcase(PurePosixPath(entry["path"]).name, pattern)
        ]
        return GlobResult(matches=matches)

    def _limit_glob_results(self, result: GlobResult) -> GlobResult:
        if result.matches is not None and len(result.matches) > self.max_glob_results:
            result.matches = result.matches[: self.max_glob_results]
            result.truncated = True
            result.error = (
                "Glob returned too many results; only the first "
                f"{self.max_glob_results} matches are returned. "
                "Narrow the directory or pattern."
            )
        return result


def _is_root_path(path: str | None) -> bool:
    if path is None:
        return True
    normalized = path.strip().replace("\\", "/").rstrip("/")
    return normalized in {"", ".", "/."}


def _is_broad_pattern(pattern: str) -> bool:
    normalized = pattern.strip().replace("\\", "/").lstrip("/")
    return normalized in {"*", "**", "*/", "**/*", "**/"} or normalized.startswith(
        "**/"
    )


def _narrow_glob_scope(
    pattern: str,
    path: str | None,
) -> tuple[str, str] | tuple[None, None]:
    normalized = pattern.strip().replace("\\", "/").lstrip("/")
    parts = normalized.split("/")
    static_parts: list[str] = []
    for part in parts:
        if any(character in part for character in "*?[{"):
            break
        static_parts.append(part)

    if not static_parts:
        return None, None

    if len(static_parts) == len(parts):
        if len(parts) == 1:
            return None, None
        static_parts = static_parts[:-1]
        remainder = parts[-1]
    else:
        remainder = "/".join(parts[len(static_parts) :])

    base_path = (
        ""
        if _is_root_path(path)
        else (path or "").strip().replace("\\", "/").rstrip("/")
    )
    scoped_path = "/".join(part for part in (base_path, *static_parts) if part)
    return scoped_path or "/", remainder


def _is_root_wide_glob(pattern: str, path: str | None) -> bool:
    return _is_root_path(path) and _is_broad_pattern(pattern)
