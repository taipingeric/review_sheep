from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


def _load_entrypoint(name: str) -> Any:
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review_chat = _load_entrypoint("review_chat")
review_pr = _load_entrypoint("review_pr")


def _clear_environment(monkeypatch: Any) -> None:
    for name in (
        "GITHUB_REPOSITORY",
        "GITHUB_TOKEN",
        "GITHUB_WORKSPACE",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "REVIEW_PR_NUMBER",
        "REVIEW_CACHE_ROOT",
        "REVIEW_CACHE_LOCK_TIMEOUT",
        "REVIEW_CACHE_WORKTREE_RETENTION",
        "REVIEW_CACHE_MIRROR_RETENTION",
        "REVIEW_CACHE_CLEANUP",
    ):
        monkeypatch.delenv(name, raising=False)


def test_chat_entrypoint_uses_process_environment_without_dotenv(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_environment(monkeypatch)
    (tmp_path / ".env").write_text(
        "GITHUB_TOKEN=dotenv-token\n"
        "OPENAI_MODEL=dotenv-model\n"
        "OPENAI_API_KEY=dotenv-key\n"
    )
    captured: dict[str, Any] = {}

    def fake_chat(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 17

    monkeypatch.setenv("GITHUB_TOKEN", "process-token")
    monkeypatch.setenv("OPENAI_MODEL", "process-model")
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")
    monkeypatch.setattr(review_chat, "run_chat", fake_chat)

    assert review_chat.main() == 17
    assert captured["config"].github_token == "process-token"
    assert captured["config"].model == "process-model"
    assert captured["config"].api_key == "process-key"


def test_chat_entrypoint_does_not_use_dotenv(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_environment(monkeypatch)
    (tmp_path / ".env").write_text(
        "GITHUB_TOKEN=dotenv-token\n"
        "OPENAI_MODEL=dotenv-model\n"
        "OPENAI_API_KEY=dotenv-key\n"
    )

    assert review_chat.main() == 2
    assert "GITHUB_TOKEN is not configured" in capsys.readouterr().err


def test_ci_entrypoint_does_not_use_dotenv(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_environment(monkeypatch)
    (tmp_path / ".env").write_text(
        "GITHUB_REPOSITORY=dotenv/repository\n"
        "GITHUB_TOKEN=dotenv-token\n"
        "REVIEW_PR_NUMBER=42\n"
        "ANTHROPIC_AUTH_TOKEN=dotenv-auth-token\n"
        "ANTHROPIC_BASE_URL=https://dotenv.example\n"
        "ANTHROPIC_DEFAULT_SONNET_MODEL=dotenv-model\n"
    )

    assert review_pr.main([]) == 2
    assert "REVIEW_PR_NUMBER is not configured" in capsys.readouterr().err


def test_ci_entrypoint_passes_process_environment_to_review_without_dotenv(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_environment(monkeypatch)
    (tmp_path / ".env").write_text(
        "GITHUB_TOKEN=dotenv-token\nANTHROPIC_AUTH_TOKEN=dotenv-auth-token\n"
    )
    monkeypatch.setenv("GITHUB_TOKEN", "process-token")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "process-auth-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://process.example")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "process-model")
    captured: dict[str, Any] = {}

    def fake_review(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 23

    monkeypatch.setattr(review_pr, "run_review", fake_review)

    assert (
        review_pr.main(
            [
                "--repo",
                "explicit/repository",
                "--pr-number",
                "42",
                "--checkout",
                str(tmp_path),
                "--instructions",
                "Focus on authorization.",
                "--model-tier",
                "sonnet",
            ]
        )
        == 23
    )
    config = captured["config"]
    assert config.repo == "explicit/repository"
    assert config.pull_request_number == 42
    assert config.github_token == "process-token"
    assert config.anthropic_auth_token == "process-auth-token"
    assert config.model == "process-model"


def test_ci_entrypoint_passes_cache_controls_to_review(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "process-token")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "process-auth-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://process.example")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "process-model")
    captured: dict[str, Any] = {}

    def fake_review(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(review_pr, "run_review", fake_review)

    assert (
        review_pr.main(
            [
                "--repo",
                "explicit/repository",
                "--pr-number",
                "42",
                "--checkout",
                str(tmp_path),
                "--cache-root",
                str(tmp_path / "cache"),
                "--cache-lock-timeout",
                "3.5",
                "--cache-worktree-retention",
                "3600",
                "--cache-mirror-retention",
                "7200",
                "--cache-cleanup",
            ]
        )
        == 0
    )
    config = captured["config"]
    assert config.cache_root == str((tmp_path / "cache").resolve())
    assert config.cache_lock_timeout == 3.5
    assert config.cache_worktree_retention == 3600
    assert config.cache_mirror_retention == 7200
    assert config.cache_cleanup is True


def test_ci_entrypoint_can_explicitly_bypass_environment_cache(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "process-token")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "process-auth-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://process.example")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "process-model")
    monkeypatch.setenv("REVIEW_CACHE_ROOT", str(tmp_path / "environment-cache"))
    captured: dict[str, Any] = {}

    def fake_review(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(review_pr, "run_review", fake_review)

    assert (
        review_pr.main(
            [
                "--repo",
                "explicit/repository",
                "--pr-number",
                "42",
                "--checkout",
                str(tmp_path),
                "--no-cache",
            ]
        )
        == 0
    )
    assert captured["config"].cache_root is None
