from __future__ import annotations

import os

from loguru import logger


def _redact(message: str) -> str:
    """Redact sensitive env values from log messages."""
    for name, value in os.environ.items():
        if not value:
            continue
        lowered = name.lower()
        if any(token in lowered for token in ("token", "secret", "password", "key")):
            message = message.replace(value, "[REDACTED]")
    return message


def _sink(message: str) -> None:
    print(_redact(message), end="")


def init_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO")
    json_logs = os.getenv("JSON_LOGS", "0") in {"1", "true", "True"}
    logger.remove()
    logger.add(_sink, level=level, serialize=json_logs)
