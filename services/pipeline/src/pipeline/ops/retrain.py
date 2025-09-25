from __future__ import annotations

"""Scheduled retrain job for ML models.

Uses environment variables:
- ENABLE_RETRAIN=1 to activate (scheduler decides when to call maybe_run)
- RETRAIN_INTERVAL_H=24 minimal hours between runs
- RETRAIN_FEATURES_S3: S3 URI to features parquet for training
- RETRAIN_HORIZONS: comma list, e.g., "4h,12h"
- ML_MODELS_S3_PREFIX: prefix to save models
"""

import os
from datetime import datetime, timezone, timedelta
from typing import List

from loguru import logger

from ..infra.db import fetch_agent_metrics, insert_agent_metric
from ..models.models_ml import train_lightgbm_randomforest, save_models_s3


def _should_run(interval_h: float) -> bool:
    rows = fetch_agent_metrics("retrain", "last_run%", limit=1)
    if not rows:
        return True
    ts_iso, _ = rows[0]
    try:
        last = datetime.fromisoformat(ts_iso)
    except Exception:
        return True
    now = datetime.now(timezone.utc)
    return (now - last) >= timedelta(hours=interval_h)


def run_once(features_s3: str, horizons: List[str]) -> tuple[str, dict]:
    export, metrics = train_lightgbm_randomforest(features_s3, horizons=horizons)
    uri = save_models_s3(export, metrics)
    return uri, metrics


def maybe_run() -> List[str]:
    if os.getenv("ENABLE_RETRAIN", "0") not in {"1", "true", "True"}:
        return []
    interval_h = float(os.getenv("RETRAIN_INTERVAL_H", "24"))
    if not _should_run(interval_h):
        return []
    features_s3 = os.getenv("RETRAIN_FEATURES_S3")
    if not features_s3:
        logger.warning("retrain: RETRAIN_FEATURES_S3 is not set; skipping")
        return []
    horizons = [
        h.strip()
        for h in os.getenv("RETRAIN_HORIZONS", "1h,4h,24h").split(",")
        if h.strip()
    ]
    if not horizons:
        horizons = ["4h"]
    uris: List[str] = []
    try:
        uri, metrics = run_once(features_s3, horizons)
        uris.append(uri)
        if isinstance(metrics, dict):
            for hz in metrics.keys():
                insert_agent_metric(
                    "retrain",
                    f"last_run_{hz}",
                    1.0,
                    labels={"horizon": hz},
                )
        else:
            insert_agent_metric("retrain", "last_run", 1.0, labels={"horizon": "aggregate"})
    except Exception as e:  # noqa: BLE001
        logger.warning(f"retrain: failed: {e}")
    return uris


def main():
    uris = maybe_run()
    for u in uris:
        print(u)


if __name__ == "__main__":
    main()
