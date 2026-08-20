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
    parser.add_argument(
        "--model-tier",
        choices=("haiku", "sonnet", "opus"),
    )
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
            model_tier=args.model_tier,
        )
    except RuntimeError as config_error:
        print(f"error: {config_error}", file=sys.stderr)
        return 2
    return run_review(config=config)


if __name__ == "__main__":
    raise SystemExit(main())
