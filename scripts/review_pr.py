"""Run one non-interactive Review for CI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from review_sheep.ci import main as run_review
from review_sheep.config import CIConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review one pull request from a fixed local Git checkout."
    )
    parser.add_argument("--repo")
    parser.add_argument("--pr-number", type=int)
    parser.add_argument("--checkout")
    parser.add_argument("--instructions")
    parser.add_argument("--lens-concurrency", type=int)
    parser.add_argument(
        "--model-tier",
        choices=("haiku", "sonnet", "opus"),
    )
    parser.add_argument("--cache-root")
    parser.add_argument(
        "--no-cache",
        action="store_const",
        const="",
        dest="cache_root",
        help="Disable the repository mirror cache for this Review.",
    )
    parser.add_argument("--cache-lock-timeout", type=float)
    parser.add_argument("--cache-worktree-retention", type=float)
    parser.add_argument("--cache-mirror-retention", type=float)
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument(
        "--cache-cleanup",
        action="store_true",
        dest="cache_cleanup",
    )
    cleanup.add_argument(
        "--no-cache-cleanup",
        action="store_false",
        dest="cache_cleanup",
    )
    parser.set_defaults(cache_cleanup=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build explicit CI configuration from trusted process inputs."""
    args = _parser().parse_args(argv)
    try:
        config = CIConfig.from_environment(
            repo=args.repo,
            pull_request_number=args.pr_number,
            checkout=args.checkout,
            instructions=args.instructions,
            lens_concurrency=args.lens_concurrency,
            model_tier=args.model_tier,
            cache_root=args.cache_root,
            cache_lock_timeout=args.cache_lock_timeout,
            cache_worktree_retention=args.cache_worktree_retention,
            cache_mirror_retention=args.cache_mirror_retention,
            cache_cleanup=args.cache_cleanup,
        )
    except (RuntimeError, ValueError) as config_error:
        print(f"error: {config_error}", file=sys.stderr)
        return 2
    return run_review(config=config)


if __name__ == "__main__":
    raise SystemExit(main())
