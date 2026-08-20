"""Run Review Sheep's interactive prompt loop."""

from __future__ import annotations

import sys

from review_sheep.chat import main
from review_sheep.config import ChatConfig

if __name__ == "__main__":
    try:
        runtime_config = ChatConfig.from_environment()
    except RuntimeError as config_error:
        print(f"error: {config_error}", file=sys.stderr)
        raise SystemExit(2) from config_error
    raise SystemExit(main(config=runtime_config))
