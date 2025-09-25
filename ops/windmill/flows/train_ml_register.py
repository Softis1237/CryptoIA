#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from loguru import logger


def _ensure_path():
    if "/app/src" not in sys.path:
        sys.path.append("/app/src")


def main():
    _ensure_path()
    # Lazy imports after path is ensured
    from pipeline.models.models_ml import train_lightgbm_randomforest, save_models_s3
    from pipeline.infra.db import upsert_model_registry

    # Resolve features S3
    features = os.getenv("FEATURES_S3")
    if not features:
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slot = os.getenv("SLOT", "manual")
        bucket = os.getenv("S3_BUCKET", "artifacts")
        features = f"s3://{bucket}/runs/{date_key}/{slot}/features.parquet"

    horizons_env = os.getenv("HORIZONS")
    horizon_minutes_env = os.getenv("HORIZON_MINUTES")
    horizons: list[str | int] = []
    if horizons_env:
        horizons = [h.strip() for h in horizons_env.split(",") if h.strip()]
    elif horizon_minutes_env:
        if "," in horizon_minutes_env:
            horizons = [int(h.strip()) for h in horizon_minutes_env.split(",") if h.strip()]
        else:
            horizons = [int(horizon_minutes_env)]
    else:
        horizons = ["4h"]
    name = os.getenv("MODEL_NAME", "sklearn-bundle")
    version = os.getenv("MODEL_VERSION") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = os.getenv("ML_MODELS_S3_PREFIX", "models/ml")

    export, metrics = train_lightgbm_randomforest(features, horizons=horizons)
    uri = save_models_s3(export, metrics, s3_key_prefix=prefix)

    # Upsert to registry
    per_h = export.get("per_horizon") or {}
    params = {
        "feature_cols": export.get("feature_cols"),
        "freq_min": export.get("freq_min"),
        "horizons": [
            {
                "label": label,
                "minutes": payload.get("horizon_minutes"),
                "models": list((payload.get("models") or {}).keys()) if isinstance(payload, dict) else [],
            }
            for label, payload in (per_h.items() if isinstance(per_h, dict) else [])
        ],
    }
    upsert_model_registry(name=name, version=version, path_s3=uri, params=params, metrics=metrics)

    out = {"name": name, "version": version, "s3_uri": uri, "metrics": metrics}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
