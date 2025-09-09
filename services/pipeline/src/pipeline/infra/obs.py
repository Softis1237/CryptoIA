from __future__ import annotations

import os
from loguru import logger


def init_sentry() -> None:
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.0,
            enable_tracing=False,
            environment=os.getenv("ENV", "dev"),
        )
        logger.info("Sentry initialized")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Sentry init failed: {e}")
