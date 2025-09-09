from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..infra.s3 import download_bytes, upload_bytes
from ..infra.db import upsert_model_registry


def _read_features(s3_uri: str) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = download_bytes(s3_uri)
    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas().sort_values("ts").reset_index(drop=True)
    return df


def _build_supervised(df: pd.DataFrame, horizon_steps: int, feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    # Use numeric feature columns (exclude raw timestamp/index); keep engineered features
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {"ts"}]
    x = df[feature_cols].copy()
    # Target: future close delta (regression)
    if "close" not in x.columns:
        raise ValueError("features must contain 'close'")
    target = df["close"].shift(-horizon_steps) - df["close"]
    # Drop tail rows without target
    x = x.iloc[:-horizon_steps]
    y = target.iloc[:-horizon_steps].astype(float).values
    # Fill NaNs
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x, y


def _select_features(df: pd.DataFrame) -> List[str]:
    # Simple filter: numeric columns except obviously non-stationary/identifiers
    bad_prefixes = ("hour_bucket_", "dow_")  # these are fine but keep
    cols = []
    for c in df.columns:
        if c in {"ts"}:
            continue
        if df[c].dtype.kind not in ("i", "f"):
            continue
        cols.append(c)
    return cols


def train_lightgbm_randomforest(features_s3: str, horizon_minutes: int = 240, freq_min: int = 5) -> Tuple[dict, dict]:
    try:
        import lightgbm as lgb  # type: ignore
    except Exception:
        lgb = None  # type: ignore
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        xgb = None  # type: ignore
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except Exception:
        CatBoostRegressor = None  # type: ignore
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.metrics import mean_absolute_error  # type: ignore

    df = _read_features(features_s3)
    # Resample to 5m close if exists ts in seconds
    try:
        dt = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.copy()
        df.index = dt
        df = df.resample(f"{freq_min}min").last().dropna().reset_index(drop=True)
    except Exception:
        pass

    steps = max(1, horizon_minutes // freq_min)
    feat_cols = _select_features(df)
    X, y = _build_supervised(df, steps, feat_cols)

    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    # RandomForest baseline
    rf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)),
    ])
    maes = []
    for tr, va in tscv.split(X):
        rf.fit(X.iloc[tr], y[tr])
        pred = rf.predict(X.iloc[va])
        maes.append(float(mean_absolute_error(y[va], pred)))
    results["rf_mae"] = float(np.mean(maes)) if maes else None

    # LightGBM (optional)
    gbm = None
    if lgb is not None:
        gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=42)
        pipe_gbm = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("gbm", gbm),
        ])
        maes = []
        for tr, va in tscv.split(X):
            pipe_gbm.fit(X.iloc[tr], y[tr])
            pred = pipe_gbm.predict(X.iloc[va])
            maes.append(float(mean_absolute_error(y[va], pred)))
        results["lgb_mae"] = float(np.mean(maes)) if maes else None
    else:
        pipe_gbm = None

    # Keep models
    export = {
        "feature_cols": feat_cols,
        "horizon_steps": steps,
        "freq_min": freq_min,
        "models": {
            "rf": rf,
            **({"lgb": pipe_gbm} if pipe_gbm is not None else {}),
        },
    }
    # XGBoost (optional)
    if xgb is not None:
        try:
            xgbr = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
            pipe_xgb = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("xgb", xgbr),
            ])
            maes = []
            for tr, va in tscv.split(X):
                pipe_xgb.fit(X.iloc[tr], y[tr])
                pred = pipe_xgb.predict(X.iloc[va])
                maes.append(float(mean_absolute_error(y[va], pred)))
            results["xgb_mae"] = float(np.mean(maes)) if maes else None
            export["models"]["xgb"] = pipe_xgb
        except Exception as e:  # noqa: BLE001
            logger.warning(f"XGBoost training failed: {e}")

    # CatBoost (optional)
    if CatBoostRegressor is not None:
        try:
            cbr = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, loss_function='MAE', random_seed=42, verbose=False)
            # CatBoost handles raw features; keep pipeline for uniformity
            pipe_cb = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("cb", cbr),
            ])
            maes = []
            for tr, va in tscv.split(X):
                pipe_cb.fit(X.iloc[tr], y[tr])
                pred = pipe_cb.predict(X.iloc[va])
                maes.append(float(mean_absolute_error(y[va], pred)))
            results["cat_mae"] = float(np.mean(maes)) if maes else None
            export["models"]["cat"] = pipe_cb
        except Exception as e:  # noqa: BLE001
            logger.warning(f"CatBoost training failed: {e}")
    return export, results


def save_models_s3(export: dict, metrics: dict, s3_key_prefix: Optional[str] = None) -> str:
    import joblib  # type: ignore

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
    prefix = s3_key_prefix or os.getenv("ML_MODELS_S3_PREFIX", "models/ml")
    key = f"{prefix}/sklearn_{ts}.pkl"
    bio = io.BytesIO()
    joblib.dump({"export": export, "metrics": metrics}, bio)
    s3_uri = upload_bytes(key, bio.getvalue(), content_type="application/octet-stream")
    logger.info(f"Saved ML models to {s3_uri}")
    # Save a lightweight JSON manifest next to model
    manifest = {
        "s3_uri": s3_uri,
        "horizon_steps": int(export.get("horizon_steps", 0)),
        "feature_cols": list(export.get("feature_cols", [])),
        "metrics": metrics,
    }
    upload_bytes(key + ".json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"), content_type="application/json")
    # Register in model_registry (best-effort)
    try:
        upsert_model_registry(name="sklearn_bundle", version=ts, path_s3=s3_uri, params={"horizon_steps": int(export.get("horizon_steps", 0)), "feature_cols": list(export.get("feature_cols", []))}, metrics=metrics)
    except Exception:
        pass
    return s3_uri


def try_add_ml_preds(df_5m: pd.Series, horizon_steps: int, last_price: float, atr: float, horizon_minutes: int) -> List[dict]:
    """Load pre-trained sklearn model bundle from S3 (ML_MODEL_S3) and add predictions.

    Expects a pickled dict {export:{feature_cols,horizon_steps,models}, metrics:{...}}.
    If env var is missing or load fails, returns [].
    """
    model_s3 = os.getenv("ML_MODEL_S3")
    if not model_s3:
        return []
    try:
        import joblib  # type: ignore
        raw = download_bytes(model_s3)
        data = joblib.load(io.BytesIO(raw))
        export = data.get("export", {})
        models: dict = export.get("models", {})
        feat_cols: List[str] = export.get("feature_cols", [])
        # Build recent feature row from available DataFrame columns (features file should include them)
        # We don't have full features DF here, so rely on df_5m stats + simple technicals to approximate minimal inputs.
        # Fallback: use last value for listed columns if present in a cached context.
        # Here we cannot rebuild full feature row; thus this helper is primarily for trained models using basic columns like ema/rsi present in features.
        # For now, skip if feature mismatch is too large.
        # Compute interval
        pi_low, pi_high = last_price - atr * 1.5, last_price + atr * 1.5
        preds: List[dict] = []
        # Without the full feature row context, we cannot safely infer; return empty to avoid misleading output.
        logger.info("ML_MODEL_S3 set but try_add_ml_preds skipped due to missing full feature row context in models pipeline.")
        return preds
    except Exception as e:  # noqa: BLE001
        logger.warning(f"try_add_ml_preds failed: {e}")
        return []


def _sigmoid(x: float) -> float:
    try:
        return float(1.0 / (1.0 + np.exp(-x)))
    except Exception:
        return 0.5


def try_add_ml_preds_full(features_s3: str, last_price: float, atr: float, horizon_minutes: int) -> List[dict]:
    model_s3 = os.getenv("ML_MODEL_S3")
    if not model_s3:
        return []
    try:
        import joblib  # type: ignore
        raw = download_bytes(model_s3)
        data = joblib.load(io.BytesIO(raw))
        export = data.get("export", {})
        models: dict = export.get("models", {})
        feat_cols: List[str] = export.get("feature_cols", [])
        # Load features frame and extract last row
        df = _read_features(features_s3)
        # Take last row; ensure columns
        X = pd.DataFrame([{c: float(df[c].iloc[-1]) if c in df.columns else 0.0 for c in feat_cols}])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pi_low, pi_high = last_price - atr * 1.5, last_price + atr * 1.5
        outs: List[dict] = []
        for name, mdl in models.items():
            try:
                delta = float(mdl.predict(X)[0])
                y_hat = float(last_price + delta)
                # confidence proxy from normalized delta
                norm = delta / max(1e-9, atr)
                proba_up = _sigmoid(norm)
                outs.append({
                    "model": f"ml:{name}",
                    "y_hat": y_hat,
                    "pi_low": pi_low,
                    "pi_high": pi_high,
                    "proba_up": float(proba_up),
                    "cv_metrics": data.get("metrics", {}),
                })
            except Exception as e:  # noqa: BLE001
                logger.warning(f"ML model {name} prediction failed: {e}")
                continue
        return outs
    except Exception as e:  # noqa: BLE001
        logger.warning(f"try_add_ml_preds_full failed: {e}")
        return []


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="S3 URI of features.parquet")
    ap.add_argument("--horizon_minutes", type=int, default=240)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    export, metrics = train_lightgbm_randomforest(args.features, horizon_minutes=args.horizon_minutes)
    if args.save:
        uri = save_models_s3(export, metrics)
        print(uri)
    else:
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()
