from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, time as dtime, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from loguru import logger

from .trade_reco_tune import run as tune_reco


def _sleep_until_next(hour_local: int = 3, minute: int = 0) -> None:
    tzname = os.getenv("TIMEZONE", "Europe/Moscow")
    tz = ZoneInfo(tzname) if ZoneInfo else None
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz) if tz else now_utc
    today = now_local.date()
    target_today = datetime.combine(today, dtime(hour_local, minute), tzinfo=tz)
    if now_local < target_today:
        target = target_today
    else:
        target = datetime.combine(today + timedelta(days=1), dtime(hour_local, minute), tzinfo=tz)
    target_utc = target.astimezone(timezone.utc)
    delta = (target_utc - now_utc).total_seconds()
    time.sleep(max(1, int(delta)))


def main() -> None:
    logger.info("rt_tune: started; will tune TradeRecommend daily")
    # Immediate run on start (best-effort)
    try:
        tune_reco()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"rt_tune: initial tune failed: {e}")
    hour = int(os.getenv("RT_TUNE_HOUR_LOCAL", "3"))
    while True:
        try:
            _sleep_until_next(hour_local=hour)
            tune_reco()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"rt_tune: tune failed: {e}")


if __name__ == "__main__":
    main()

