from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..infra.s3 import download_bytes, upload_bytes
from ..infra.db import upsert_model_registry
from ..utils.horizons import horizon_to_minutes, minutes_to_horizon, normalize_horizons


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


def _train_single_horizon(
    df: pd.DataFrame,
    horizon_minutes: int,
    freq_min: int,
    feature_cols: List[str],
    *,
    lgb_module=None,
    xgb_module=None,
    cat_cls=None,
) -> Tuple[Dict[str, object], Dict[str, float], int]:
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.metrics import mean_absolute_error  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    steps = max(1, horizon_minutes // freq_min)
    X, y = _build_supervised(df, steps, feature_cols)

    n_samples = len(X)
    tscv = None
    if n_samples >= max(40, steps * 3):
        try:
            n_splits = min(5, max(2, n_samples // max(steps, 15)))
            n_splits = max(2, min(n_splits, n_samples - 1))
            if n_splits >= 2 and n_samples > n_splits:
                tscv = TimeSeriesSplit(n_splits=n_splits)
        except Exception:
            tscv = None

    def _fit_and_score(model: Pipeline, name: str) -> float | None:
        maes: List[float] = []
        if tscv is not None:
            for tr_idx, va_idx in tscv.split(X):
                model.fit(X.iloc[tr_idx], y[tr_idx])
                try:
                    pred = model.predict(X.iloc[va_idx])
                except Exception:
                    pred = np.zeros_like(y[va_idx])
                try:
                    maes.append(float(mean_absolute_error(y[va_idx], pred)))
                except Exception:
                    continue
        else:
            try:
                model.fit(X, y)
                pred_full = model.predict(X)
                maes.append(float(mean_absolute_error(y, pred_full)))
            except Exception:
                return None
        try:
            model.fit(X, y)
        except Exception:
            return None
        return float(np.mean(maes)) if maes else None

    results: Dict[str, float | None] = {}
    models: Dict[str, object] = {}

    rf = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=9,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    results["rf_mae"] = _fit_and_score(rf, "rf")
    models["rf"] = rf

    if lgb_module is not None:
        try:
            gbm = lgb_module.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            pipe_gbm = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("gbm", gbm),
            ])
            results["lgb_mae"] = _fit_and_score(pipe_gbm, "lgb")
            models["lgb"] = pipe_gbm
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LightGBM training skipped: {exc}")

    if xgb_module is not None:
        try:
            xgbr = xgb_module.XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            pipe_xgb = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("xgb", xgbr),
            ])
            results["xgb_mae"] = _fit_and_score(pipe_xgb, "xgb")
            models["xgb"] = pipe_xgb
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"XGBoost training failed: {exc}")

    if cat_cls is not None:
        try:
            cbr = cat_cls(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                loss_function="MAE",
                random_seed=42,
                verbose=False,
            )
            pipe_cb = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("cb", cbr),
            ])
            results["cat_mae"] = _fit_and_score(pipe_cb, "cat")
            models["cat"] = pipe_cb
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"CatBoost training failed: {exc}")

    return models, results, steps


def _resolve_horizon_label(
    per_horizon: Mapping[str, Mapping[str, object]],
    horizon_minutes: int,
    default_label: Optional[str] = None,
) -> Optional[str]:
    if not per_horizon:
        return default_label
    target_minutes = max(1, int(round(float(horizon_minutes or 0))))
    best_label: Optional[str] = None
    best_diff: Optional[float] = None
    for label, payload in per_horizon.items():
        try:
            payload_minutes = int(
                float(payload.get("horizon_minutes") or horizon_to_minutes(label))
            )
        except Exception:
            payload_minutes = target_minutes
        diff = abs(payload_minutes - target_minutes)
        if best_diff is None or diff < best_diff:
            best_label = label
            best_diff = diff
    return best_label or default_label


def train_lightgbm_randomforest(
    features_s3: str,
    horizons: Iterable[str | int] | None = None,
    freq_min: int = 5,
) -> Tuple[dict, dict]:
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

    df = _read_features(features_s3)
    try:
        dt = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.copy()
        df.index = dt
        df = df.resample(f"{freq_min}min").last().dropna().reset_index(drop=True)
    except Exception:
        df = df.copy()

    feat_cols = _select_features(df)

    if horizons is None:
        requested = ["4h"]
    else:
        requested: List[str] = []
        for hz in horizons:
            if isinstance(hz, (int, float)):
                minutes_val = int(round(float(hz)))
                if minutes_val <= 0:
                    continue
                try:
                    requested.append(minutes_to_horizon(minutes_val))
                except Exception:
                    requested.append(f"{minutes_val}m")
            else:
                requested.append(str(hz))
        if not requested:
            requested = ["4h"]
    horizon_labels = normalize_horizons(requested)
    if not horizon_labels:
        horizon_labels = ["4h"]

    per_horizon: Dict[str, Dict[str, object]] = {}
    metrics_bundle: Dict[str, Dict[str, float | None]] = {}

    for label in horizon_labels:
        try:
            minutes = horizon_to_minutes(label)
        except Exception:
            try:
                minutes = horizon_to_minutes(minutes_to_horizon(int(float(label))))
            except Exception:
                logger.warning(f"Unsupported horizon label {label}; skipping")
                continue

        models, metrics, steps = _train_single_horizon(
            df,
            minutes,
            freq_min,
            feat_cols,
            lgb_module=lgb,
            xgb_module=xgb,
            cat_cls=CatBoostRegressor,
        )
        per_horizon[label] = {
            "horizon_minutes": minutes,
            "horizon_steps": steps,
            "models": models,
        }
        metrics_bundle[label] = metrics

    export: Dict[str, object] = {
        "feature_cols": feat_cols,
        "freq_min": freq_min,
        "per_horizon": per_horizon,
    }

    if per_horizon:
        default_label = horizon_labels[0]
        if default_label in per_horizon:
            export["default_horizon"] = default_label
            export["horizon_steps"] = per_horizon[default_label]["horizon_steps"]
            export["models"] = per_horizon[default_label]["models"]
            export["horizon_minutes"] = per_horizon[default_label]["horizon_minutes"]

    return export, metrics_bundle


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
    per_horizon = export.get("per_horizon") or {}
    manifest_horizons: List[Dict[str, object]] = []
    if isinstance(per_horizon, dict):
        for label, payload in per_horizon.items():
            if not isinstance(payload, dict):
                continue
            models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
            manifest_horizons.append(
                {
                    "label": label,
                    "minutes": int(payload.get("horizon_minutes", 0) or 0),
                    "models": list((models or {}).keys()),
                    "metrics": metrics.get(label) if isinstance(metrics, dict) else None,
                }
            )
    manifest = {
        "s3_uri": s3_uri,
        "feature_cols": list(export.get("feature_cols", [])),
        "freq_min": export.get("freq_min"),
        "default_horizon": export.get("default_horizon"),
        "horizons": manifest_horizons,
    }
    upload_bytes(key + ".json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"), content_type="application/json")
    # Register in model_registry (best-effort)
    try:
        upsert_model_registry(
            name="sklearn_bundle",
            version=ts,
            path_s3=s3_uri,
            params={
                "feature_cols": list(export.get("feature_cols", [])),
                "freq_min": export.get("freq_min"),
                "default_horizon": export.get("default_horizon"),
                "horizons": manifest_horizons,
            },
            metrics=metrics,
        )
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


def try_add_ml_preds_full(
    features_s3: str,
    last_price: float,
    atr: float,
    horizon_minutes: int,
) -> List[dict]:
    model_s3 = os.getenv("ML_MODEL_S3")
    if not model_s3:
        return []
    try:
        import joblib  # type: ignore

        raw = download_bytes(model_s3)
        data = joblib.load(io.BytesIO(raw))
        export = data.get("export", {}) or {}
        metrics = data.get("metrics", {}) or {}
        per_horizon = export.get("per_horizon") or {}
        default_label = export.get("default_horizon")
        label = _resolve_horizon_label(per_horizon, horizon_minutes, default_label)
        bundle = per_horizon.get(label) if isinstance(per_horizon, dict) else None
        if bundle and isinstance(bundle, dict):
            models = bundle.get("models", {}) or {}
            hz_minutes = int(bundle.get("horizon_minutes") or horizon_minutes or 240)
        else:
            models = export.get("models", {}) or {}
            try:
                hz_minutes = int(export.get("horizon_minutes") or horizon_minutes or 240)
            except Exception:
                hz_minutes = int(horizon_minutes or 240)
            if not label:
                try:
                    label = minutes_to_horizon(hz_minutes)
                except Exception:
                    label = None

        if not models:
            return []

        feat_cols: List[str] = list(export.get("feature_cols", []))
        if not feat_cols:
            logger.info("ML bundle missing feature_cols; skipping ML predictions")
            return []

        df = _read_features(features_s3)
        if df.empty:
            return []
        row = {c: float(df[c].iloc[-1]) if c in df.columns else 0.0 for c in feat_cols}
        X = pd.DataFrame([row]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        hz_minutes = max(1, int(hz_minutes))
        band_scale = max(1.0, float(np.sqrt(hz_minutes / 60.0)))
        band = float(atr * 1.5 * band_scale)
        pi_low, pi_high = last_price - band, last_price + band
        horizon_metrics = metrics.get(label) if isinstance(metrics, dict) else metrics

        outs: List[dict] = []
        for name, mdl in models.items():
            try:
                delta = float(mdl.predict(X)[0])
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"ML model {name} prediction failed: {exc}")
                continue
            y_hat = float(last_price + delta)
            norm = delta / max(1e-9, atr)
            proba_up = _sigmoid(norm)
            outs.append(
                {
                    "model": f"ml:{name}{('@' + label) if label else ''}",
                    "y_hat": y_hat,
                    "pi_low": pi_low,
                    "pi_high": pi_high,
                    "proba_up": float(proba_up),
                    "cv_metrics": horizon_metrics or {},
                    "horizon": label,
                }
            )
        return outs
    except Exception as e:  # noqa: BLE001
        logger.warning(f"try_add_ml_preds_full failed: {e}")
        return []


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="S3 URI of features.parquet")
    ap.add_argument("--horizons", nargs="+", help="Forecast horizons like 1h 4h 24h")
    ap.add_argument(
        "--horizon_minutes",
        type=int,
        nargs="+",
        help="Legacy horizon list in minutes",
    )
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    horizons: List[str | int] = []
    if args.horizons:
        horizons.extend(args.horizons)
    if args.horizon_minutes:
        horizons.extend(args.horizon_minutes)

    export, metrics = train_lightgbm_randomforest(
        args.features,
        horizons=horizons or None,
    )
    if args.save:
        uri = save_models_s3(export, metrics)
        print(uri)
    else:
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()
