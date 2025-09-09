from __future__ import annotations

import json
import os
from loguru import logger


def init_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO")
    json_logs = os.getenv("JSON_LOGS", "0") in {"1", "true", "True"}
    logger.remove()
    if json_logs:
        logger.add(lambda msg: print(msg, end=""), level=level, serialize=True)
    else:
        logger.add(lambda msg: print(msg, end=""), level=level)

