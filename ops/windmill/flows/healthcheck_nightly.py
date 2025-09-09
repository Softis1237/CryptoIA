#!/usr/bin/env python3
from __future__ import annotations

import sys
from loguru import logger


def main():
    try:
        if "/app/src" not in sys.path:
            sys.path.append("/app/src")
        from pipeline.infra.healthcheck import run as hc  # type: ignore

        ok = hc()
        if not ok:
            raise SystemExit(1)
        logger.info("Healthcheck OK")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Healthcheck failed: {e}")
        raise


if __name__ == "__main__":
    main()
