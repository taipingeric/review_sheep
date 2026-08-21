"""Run Review Sheep's interactive prompt loop."""

from __future__ import annotations

import sys

from review_sheep.chat import main as run_chat
from review_sheep.config import ChatConfig


def main() -> int:
    """Build interactive Chat configuration from trusted process inputs."""
    try:
        runtime_config = ChatConfig.from_environment()
    except RuntimeError as config_error:
        print(f"error: {config_error}", file=sys.stderr)
        return 2
    return run_chat(config=runtime_config)


if __name__ == "__main__":
    raise SystemExit(main())
