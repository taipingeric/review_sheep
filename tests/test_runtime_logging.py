from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from uuid import uuid4

import pytest

from review_sheep import runtime_logging
from review_sheep.runtime_logging import configure_console_logging


def _owned_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [
        handler
        for handler in logger.handlers
        if getattr(handler, "_review_sheep_console", False)
    ]


@pytest.fixture(autouse=True)
def project_logger(monkeypatch: pytest.MonkeyPatch) -> Iterator[logging.Logger]:
    logger_name = f"review_sheep.tests.logging.{uuid4().hex}"
    logger = logging.getLogger(logger_name)
    monkeypatch.setattr(runtime_logging, "_LOGGER_NAME", logger_name)
    try:
        yield logger
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def test_configure_console_logging_emits_formatted_logs_at_requested_level(
    project_logger: logging.Logger,
) -> None:
    logger = project_logger
    stream = io.StringIO()

    configure_console_logging(stream=stream, level=" debug ")
    logger.debug("debug message")

    output = stream.getvalue()
    assert f"DEBUG [{logger.name}] debug message" in output
    assert re.match(r"^\d{2}:\d{2}:\d{2} ", output)
    assert logger.level == logging.DEBUG
    assert logger.propagate is False


def test_configure_console_logging_replaces_owned_handler_and_falls_back_to_info(
    project_logger: logging.Logger,
) -> None:
    logger = project_logger
    first_stream = io.StringIO()
    second_stream = io.StringIO()
    unrelated_handler = logging.NullHandler()
    logger.addHandler(unrelated_handler)

    configure_console_logging(stream=first_stream, level="DEBUG")
    first_handler = _owned_handlers(logger)[0]
    configure_console_logging(stream=second_stream, level="not-a-level")
    assert logger.level == logging.INFO
    assert len(_owned_handlers(logger)) == 1
    assert unrelated_handler in logger.handlers
    assert first_handler not in logger.handlers

    logger.debug("hidden debug")
    logger.info("visible info")

    assert "hidden debug" not in second_stream.getvalue()
    assert f"INFO [{logger.name}] visible info" in second_stream.getvalue()
    assert first_stream.getvalue() == ""
