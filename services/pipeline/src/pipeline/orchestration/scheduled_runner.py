from __future__ import annotations

import os
import time

from loguru import logger

from ..infra.db import get_conn
from ..infra.health import start_background as start_health_server
from ..infra.logging_config import init_logging
from ..ops.cv_metrics_push import run as push_cv_metrics
from ..ops.feedback_metrics import collect_outcomes, update_regime_alphas
from ..ops.retrain import maybe_run as maybe_retrain
from .agent_flow import run_release_flow


def main():
    init_logging()
    start_health_server()
    slot = os.environ.get("SLOT", "scheduled")
    interval = int(os.environ.get("RUN_INTERVAL_SEC", "3600"))
    logger.info(f"Scheduled runner started: slot={slot} interval={interval}s")
    while True:
        try:
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
        except Exception as e:  # noqa: BLE001
            logger.exception(f"scheduled_runner: run failed: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
