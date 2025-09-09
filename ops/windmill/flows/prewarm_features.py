#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from loguru import logger


def main(hours_back: int = 72):
    try:
        if "/app/src" not in sys.path:
            sys.path.append("/app/src")
        from pipeline.data.ingest_prices import IngestPricesInput, run as run_prices  # type: ignore
        from pipeline.features.features_calc import FeaturesCalcInput, run as run_features  # type: ignore

        now = datetime.now(timezone.utc)
        end_ts = int(now.timestamp())
        start_ts = end_ts - hours_back * 3600
        slot = os.getenv("SLOT", "prewarm")
        run_id = now.strftime("prewarm-%Y%m%dT%H%M%SZ")
        p_in = IngestPricesInput(run_id=run_id, slot=slot, symbols=["BTCUSDT"], start_ts=start_ts, end_ts=end_ts)
        p_out = run_prices(p_in)
        f_in = FeaturesCalcInput(prices_path_s3=p_out.prices_path_s3, news_signals=[], run_id=run_id, slot=slot)
        _ = run_features(f_in)
        logger.info("Prewarm complete")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Prewarm failed: {e}")
        raise


if __name__ == "__main__":
    main()
