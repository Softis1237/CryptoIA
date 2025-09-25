from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.arima.model import ARIMA

from ..infra.s3 import download_bytes
from .models_darts import try_add_darts_preds  # optional, handles import errors
from .models_neuralprophet import try_add_np_preds  # optional
from .models_prophet import try_add_prophet_preds  # optional
from .models_ml import try_add_ml_preds_full  # optional


class ModelsInput(BaseModel):
    features_path_s3: str
    regime_label: Optional[str] = None
    neighbors: Optional[list] = None
    horizon_minutes: int = 240  # 4h default


class ModelPred(BaseModel):
    model: str
    y_hat: float
    pi_low: float
    pi_high: float
    proba_up: float
    cv_metrics: dict


class ModelsOutput(BaseModel):
    preds: List[ModelPred]
    last_price: float
    atr: float
    atr_map: Dict[str, float] = Field(default_factory=dict)


def _load_prices(features_path_s3: str) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = download_bytes(features_path_s3)
    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas()
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("dt").sort_index()
    return df


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return float(100.0 * np.mean(np.abs(y_pred - y_true) / denom))


def _exponential_smoothing_forecast(series: pd.Series, steps: int) -> np.ndarray:
    # Simple ETS (no seasonality for minute data)
    model = HWES(series, trend="add", seasonal=None, initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(steps)
    return fc.values.astype(float)


def _naive_forecast(series: pd.Series, steps: int) -> np.ndarray:
    return np.repeat(series.iloc[-1], steps)


def _interval_from_vol(last_price: float, atr: float, horizon_minutes: int) -> tuple[float, float]:
    # scale ATR by sqrt(time)
    scale = max(1.0, np.sqrt(horizon_minutes / 60.0))
    band = float(atr * 1.5 * scale)
    return last_price - band, last_price + band


def _arima_forecast(series: pd.Series, steps: int, order=(1, 1, 1)) -> np.ndarray:
    try:
        model = ARIMA(series, order=order)
        fit = model.fit()
        fc = fit.forecast(steps)
        return np.asarray(fc, dtype=float)
    except Exception:
        # Fallback to naive if ARIMA fails
        return _naive_forecast(series, steps)


def _backtest_one_step(series: pd.Series, model_fn, window: int = 60) -> tuple[float, float]:
    n = len(series)
    y_true = []
    y_pred = []
    y_prev = []
    # iterate forward over last `window` points
    for idx in range(max(1, n - window), n):
        prev = series.iloc[idx - 1] if idx - 1 >= 0 else series.iloc[0]
        train = series.iloc[:idx]
        pred = model_fn(train, 1)[-1]
        y_prev.append(float(prev))
        y_true.append(float(series.iloc[idx]))
        y_pred.append(float(pred))
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    yp_prev = np.asarray(y_prev, dtype=float)
    smape = _smape(yt, yp)
    da = float(np.mean(np.sign(yt - yp_prev) == np.sign(yp - yp_prev))) if len(yt) > 0 else 0.0
    return smape, da


def run(payload: ModelsInput) -> ModelsOutput:
    df = _load_prices(payload.features_path_s3)
    # Ensure numeric types
    for col in ["close", "atr_14"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).copy()

    # Resample to 5-min to reduce noise/compute steps correctly
    close_5m = df["close"].resample("5min").last().dropna()
    atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else float(np.std(close_5m.diff().dropna()))
    atr_map: Dict[str, float] = {}
    for label in ("30m", "1h", "4h", "12h", "24h"):
        col = f"atr_{label}"
        if col in df.columns:
            try:
                atr_map[label] = float(df[col].iloc[-1])
            except Exception:
                continue
    if "atr_dynamic" in df.columns:
        try:
            atr_map.setdefault("dynamic", float(df["atr_dynamic"].iloc[-1]))
        except Exception:
            pass
    atr_map.setdefault("baseline", atr)
    last_price = float(close_5m.iloc[-1])

    horizon_steps = max(1, payload.horizon_minutes // 5)

    preds: List[ModelPred] = []

    # ETS model
    try:
        fc_ets = _exponential_smoothing_forecast(close_5m, horizon_steps)
        y_hat_ets = float(fc_ets[-1])
        pi_low, pi_high = _interval_from_vol(y_hat_ets, atr, payload.horizon_minutes)
        # simple cv metric: compare one-step-ahead on last 60 steps
        smape, da = _backtest_one_step(close_5m, lambda s, n: _exponential_smoothing_forecast(s, n))
        proba_up = 0.5 + np.tanh((y_hat_ets - last_price) / (atr + 1e-8)) * 0.25
        preds.append(ModelPred(model="ets", y_hat=y_hat_ets, pi_low=pi_low, pi_high=pi_high, proba_up=float(proba_up), cv_metrics={"smape": smape, "da": da}))
    except Exception as e:  # noqa: BLE001
        logger.exception(f"ETS failed: {e}")

    # Naive (persistence)
    try:
        fc_nv = _naive_forecast(close_5m, horizon_steps)
        y_hat_nv = float(fc_nv[-1])
        pi_low, pi_high = _interval_from_vol(y_hat_nv, atr, payload.horizon_minutes)
        smape, da = _backtest_one_step(close_5m, lambda s, n: _naive_forecast(s, n))
        proba_up = 0.5  # neutral
        preds.append(ModelPred(model="naive", y_hat=y_hat_nv, pi_low=pi_low, pi_high=pi_high, proba_up=float(proba_up), cv_metrics={"smape": smape, "da": da}))
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Naive failed: {e}")

    # ARIMA model (optionally tuned)
    try:
        order = (1, 1, 1)
        try:
            import os as _os
            if _os.getenv("AUTO_TUNE_ARIMA", "0") in {"1", "true", "True"}:
                from .tuning import tune_arima as _tune_arima  # lazy import

                order, best_smape = _tune_arima(close_5m, horizon_steps=horizon_steps)
                logger.info(f"ARIMA tuned order={order} smape={best_smape:.3f}")
        except Exception as _e:  # noqa: BLE001
            logger.warning(f"ARIMA tuning skipped: {_e}")

        fc_arima = _arima_forecast(close_5m, horizon_steps, order=order)
        y_hat_arima = float(fc_arima[-1])
        pi_low, pi_high = _interval_from_vol(y_hat_arima, atr, payload.horizon_minutes)
        smape, da = _backtest_one_step(close_5m, lambda s, n, _order=order: _arima_forecast(s, n, order=_order))
        proba_up = 0.5 + np.tanh((y_hat_arima - last_price) / (atr + 1e-8)) * 0.25
        preds.append(
            ModelPred(
                model="arima",
                y_hat=y_hat_arima,
                pi_low=pi_low,
                pi_high=pi_high,
                proba_up=float(proba_up),
                cv_metrics={"smape": smape, "da": da, "order": str(order)},
            )
        )
    except Exception as e:  # noqa: BLE001
        logger.exception(f"ARIMA failed: {e}")

    # Optional Darts (Theta)
    try:
        for p in try_add_darts_preds(close_5m, horizon_steps, last_price, atr, payload.horizon_minutes):
            preds.append(ModelPred(**p))
    except Exception:
        pass
    # Optional NeuralProphet
    try:
        for p in try_add_np_preds(close_5m, horizon_steps, last_price, atr, payload.horizon_minutes):
            preds.append(ModelPred(**p))
    except Exception:
        pass
    # Optional Prophet
    try:
        for p in try_add_prophet_preds(close_5m, horizon_steps, last_price, atr, payload.horizon_minutes):
            preds.append(ModelPred(**p))
    except Exception:
        pass

    # Optional ML (LightGBM/RandomForest via sklearn bundle loaded from S3)
    try:
        for p in try_add_ml_preds_full(payload.features_path_s3, last_price, atr, payload.horizon_minutes):
            preds.append(ModelPred(**p))
    except Exception:
        pass

    return ModelsOutput(preds=preds, last_price=last_price, atr=atr, atr_map=atr_map)
