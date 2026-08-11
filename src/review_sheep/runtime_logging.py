"""Console logging configuration for Review Sheep entrypoints."""

from __future__ import annotations

import logging
from typing import TextIO

_LOGGER_NAME = "review_sheep"


def configure_console_logging(*, stream: TextIO, level: str = "INFO") -> None:
    """Emit project logs to one entrypoint-owned console stream."""
    resolved_level = getattr(logging, level.strip().upper(), logging.INFO)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(resolved_level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_review_sheep_console", False):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(stream)
    handler.setLevel(resolved_level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler._review_sheep_console = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
