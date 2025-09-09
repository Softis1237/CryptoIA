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

    horizon_minutes = int(os.getenv("HORIZON_MINUTES", "240"))
    name = os.getenv("MODEL_NAME", "sklearn-bundle")
    version = os.getenv("MODEL_VERSION") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = os.getenv("ML_MODELS_S3_PREFIX", "models/ml")

    export, metrics = train_lightgbm_randomforest(features, horizon_minutes=horizon_minutes)
    uri = save_models_s3(export, metrics, s3_key_prefix=prefix)

    # Upsert to registry
    params = {
        "horizon_minutes": horizon_minutes,
        "feature_cols": export.get("feature_cols"),
        "freq_min": export.get("freq_min"),
        "models": list((export.get("models") or {}).keys()),
    }
    upsert_model_registry(name=name, version=version, path_s3=uri, params=params, metrics=metrics)

    out = {"name": name, "version": version, "s3_uri": uri, "metrics": metrics}
    print(json.dumps(out))


if __name__ == "__main__":
    main()

