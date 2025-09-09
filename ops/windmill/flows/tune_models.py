#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from loguru import logger


def main():
    if "/app/src" not in sys.path:
        sys.path.append("/app/src")
    try:
        from pipeline.models.tuning import main as tune_main  # type: ignore
    except Exception as e:  # noqa: BLE001
        logger.error(f"Import failed: {e}")
        raise
    # Use latest features path in artifacts, or expect env FEATURES_S3
    features = os.getenv("FEATURES_S3")
    if not features:
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slot = os.getenv("SLOT", "manual")
        features = f"s3://{os.getenv('S3_BUCKET','artifacts')}/runs/{date_key}/{slot}/features.parquet"
    sys.argv = [sys.argv[0], "--features", features]
    tune_main()


if __name__ == "__main__":
    main()
