#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone


def _ensure_path():
    if "/app/src" not in sys.path:
        sys.path.append("/app/src")


def main():
    _ensure_path()
    from pipeline.agents.pattern_discovery import DiscoveryInput, run as run_discovery

    payload = DiscoveryInput(
        symbol=os.getenv("DISC_SYMBOL", "BTC/USDT"),
        provider=os.getenv("DISC_PROVIDER", "binance"),
        timeframe=os.getenv("DISC_TIMEFRAME", "1h"),
        days=int(os.getenv("DISC_DAYS", "120")),
        move_threshold=float(os.getenv("DISC_MOVE_THRESHOLD", "0.05")),
        window_hours=int(os.getenv("DISC_WINDOW_HOURS", "24")),
        lookback_hours_pre=int(os.getenv("DISC_LOOKBACK_PRE_HOURS", "48")),
        sample_limit=int(os.getenv("DISC_SAMPLE_LIMIT", "40")),
        dry_run=os.getenv("DISC_DRY_RUN", "1") in {"1", "true", "True"},
    )
    res = run_discovery(payload)
    print(json.dumps(res))


if __name__ == "__main__":
    main()

