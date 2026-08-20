"""Run Review Sheep's interactive prompt loop."""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from review_sheep.chat import main
from review_sheep.config import ChatConfig

if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    try:
        config = ChatConfig.from_environment()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    raise SystemExit(main(config=config))
