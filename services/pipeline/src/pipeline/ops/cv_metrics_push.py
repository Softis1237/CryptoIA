from __future__ import annotations

"""Push a snapshot of recent CV metrics from DB to Prometheus Pushgateway.

Exports last known values for sMAPE/MAE/DA per horizon into `pipeline_value` under
names: cv_smape_4h, cv_smape_12h, cv_mae_4h, cv_mae_12h, cv_da_4h, cv_da_12h.

Use as a lightweight cron or call from orchestration.
"""

from typing import Dict

from loguru import logger

from ..infra.db import fetch_agent_metrics
from ..infra.metrics import push_values


def _last_value(agent: str, metric_like: str) -> float | None:
    try:
        rows = fetch_agent_metrics(agent, metric_like, limit=1)
        if rows:
            return float(rows[0][1])
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"cv_metrics_push: fetch failed agent={agent} metric_like={metric_like}: {e}")
        return None


def run() -> Dict[str, float]:
    vals: Dict[str, float] = {}
    pairs = [
        ("cv_smape_4h", ("models_cv", "smape_4h%")),
        ("cv_smape_12h", ("models_cv", "smape_12h%")),
        ("cv_mae_4h", ("models_cv", "mae_4h%")),
        ("cv_mae_12h", ("models_cv", "mae_12h%")),
        ("cv_da_4h", ("models_cv", "da_4h%")),
        ("cv_da_12h", ("models_cv", "da_12h%")),
    ]
    for name, (agent, like) in pairs:
        v = _last_value(agent, like)
        if v is not None:
            vals[name] = float(v)
    if vals:
        push_values(job="models_cv", values=vals, labels={})
        logger.info(f"cv_metrics_push: pushed {vals}")
    else:
        logger.info("cv_metrics_push: nothing to push")
    return vals


def main():
    run()


if __name__ == "__main__":
    main()

