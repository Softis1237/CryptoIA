from __future__ import annotations

import os
import time

from loguru import logger
from datetime import datetime, timedelta, time as dtime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - py<3.9 fallback not expected here
    ZoneInfo = None  # type: ignore

from ..infra.db import get_conn
from ..infra.health import start_background as start_health_server
from ..infra.logging_config import init_logging
from ..ops.cv_metrics_push import run as push_cv_metrics
from ..ops.feedback_metrics import collect_outcomes, update_regime_alphas
from ..ops.retrain import maybe_run as maybe_retrain
from ..ops.rt_paper_adapt import run as paper_adapt
from .agent_flow import run_release_flow
from ..agents.master import run_master_flow
from ..mcp.server import serve_in_thread as _mcp_serve_in_thread


def _seconds_to_next_release(tz_name: str = "Europe/Moscow") -> int:
    tz = None
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz) if tz else now_utc
    # Two releases per day: 00:00 and 12:00 local time
    today = now_local.date()
    candidates = [
        datetime.combine(today, dtime(0, 0), tzinfo=tz),
        datetime.combine(today, dtime(12, 0), tzinfo=tz),
        datetime.combine(today + timedelta(days=1), dtime(0, 0), tzinfo=tz),
        datetime.combine(today + timedelta(days=1), dtime(12, 0), tzinfo=tz),
    ]
    future = [c for c in candidates if c > now_local]
    target_local = min(future) if future else candidates[-1]
    target_utc = target_local.astimezone(timezone.utc)
    delta = (target_utc - now_utc).total_seconds()
    return max(1, int(delta))


def main():
    init_logging()
    start_health_server()
    # Start MCP mini-server if enabled
    try:
        if os.getenv("ENABLE_MCP", "0") in {"1", "true", "True"}:
            _mcp_serve_in_thread(host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8765")))
    except Exception:
        logger.warning("Failed to start MCP mini-server; continuing without it")
    slot = os.environ.get("SLOT", "scheduled")
    interval = int(os.environ.get("RUN_INTERVAL_SEC", "3600"))
    twice_daily = os.environ.get("RUN_TWICE_DAILY", "1") in {"1", "true", "True"}
    tz = os.environ.get("TIMEZONE", "Europe/Moscow")
    logger.info(
        f"Scheduled runner started: slot={slot} mode={'twice-daily' if twice_daily else f'interval {interval}s'} tz={tz}"
    )
    while True:
        try:
            if os.getenv("USE_MASTER_AGENT", "0") in {"1", "true", "True"}:
                run_master_flow(slot=slot)
            else:
                run_release_flow(slot=slot)
            # Push recent CV metrics snapshot for alerts/observability
            try:
                push_cv_metrics()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"scheduled_runner: cv_metrics_push failed: {e}")
            # Collect realized outcomes after each cycle (safe, best-effort)
            try:
                collect_outcomes("4h")
                collect_outcomes("12h")
                # RT horizons
                rt_hz = os.getenv("RT_OUTCOMES_HORIZONS", "30m,60m").replace(" ", ",").split(",")
                for hz in [h.strip() for h in rt_hz if h.strip()]:
                    try:
                        collect_outcomes(hz)
                    except Exception:
                        pass
            except Exception as e:  # noqa: BLE001
                logger.warning(f"scheduled_runner: collect_outcomes failed: {e}")
            # Update regime alphas not more often than ALPHA_UPDATE_INTERVAL_H
            try:
                interval_h = int(os.getenv("ALPHA_UPDATE_INTERVAL_H", "6"))
                do_update = True
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT max(updated_at) FROM model_trust_regime")
                        row = cur.fetchone()
                        if row and row[0] is not None:
                            from datetime import datetime, timezone

                            last = row[0]
                            now = datetime.now(timezone.utc)
                            age_h = (now - last).total_seconds() / 3600.0
                            do_update = age_h >= max(1.0, float(interval_h))
                if do_update:
                    update_regime_alphas()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"scheduled_runner: update_regime_alphas failed: {e}")
            # Optionally retrain on schedule
            try:
                maybe_retrain()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"scheduled_runner: retrain failed: {e}")
            # Adapt model trust from paper PnL (best-effort)
            try:
                if os.getenv("RUN_PAPER_ADAPT", "1") in {"1", "true", "True"}:
                    paper_adapt()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"scheduled_runner: paper_adapt failed: {e}")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"scheduled_runner: run failed: {e}")
        try:
            if twice_daily:
                sleep_for = _seconds_to_next_release(tz)
            else:
                sleep_for = interval
        except Exception:
            sleep_for = interval
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
