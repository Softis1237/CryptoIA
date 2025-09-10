from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import load
from loguru import logger

from ..infra.s3 import download_bytes
from .regime_detect import detect as heuristic_detect


@dataclass
class RegimeProbs:
    label: str
    proba: Dict[str, float]
    features: Dict[str, float]


def _load_model() -> object | None:
    """Try loading a pre-trained classifier from local path or S3."""
    path = os.getenv("REGIME_MODEL_PATH")
    path_s3 = os.getenv("REGIME_MODEL_S3")

    if path_s3:
        try:
            raw = download_bytes(path_s3)
            return load(io.BytesIO(raw))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"regime.predictor_ml: failed to load model from S3: {e}")

    if path and os.path.exists(path):
        try:
            return load(path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"regime.predictor_ml: failed to load model from {path}: {e}")

    return None


def _extract_features(df: pd.DataFrame) -> tuple[np.ndarray, Dict[str, float]]:
    close = df["close"].astype(float)
    ema20 = df.get("ema_20", close.ewm(span=20, adjust=False).mean()).astype(float)
    ema50 = df.get("ema_50", close.ewm(span=50, adjust=False).mean()).astype(float)
    atr = (
        df.get("atr_14").astype(float)
        if "atr_14" in df.columns
        else (df["high"] - df["low"]).rolling(14).mean().fillna(method="bfill").astype(float)
    )

    trend = float(((ema20 - ema50) / (close.abs() + 1e-9)).ewm(span=20, adjust=False).mean().iloc[-1])
    vol = float((atr.iloc[-1] / max(1e-9, close.iloc[-1])) * 100.0)
    try:
        ret_30 = float(close.pct_change(30).fillna(0.0).iloc[-1] * 100.0)
    except Exception:
        ret_30 = 0.0

    feats = np.array([[trend, vol, ret_30]], dtype=float)
    return feats, {"trend": trend, "vol_pct": vol, "ret_30m_pct": ret_30}


def predict(features_path_s3: str) -> RegimeProbs:
    """Predict market regime using a trained ML classifier.

    If no model is available, fall back to heuristic detection.
    """
    try:
        raw = download_bytes(features_path_s3)
        table = pq.read_table(pa.BufferReader(raw))
        df = table.to_pandas().sort_values("ts").reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"regime.predictor_ml: failed to read features, fallback to heuristic: {e}"
        )
        h = heuristic_detect(features_path_s3)
        return RegimeProbs(
            label=h.label,
            proba={h.label: h.confidence},
            features=h.features,
        )

    feats, feat_dict = _extract_features(df)
    model = _load_model()
    if model is None:
        logger.warning("regime.predictor_ml: model unavailable, using heuristic")
        h = heuristic_detect(features_path_s3)
        return RegimeProbs(
            label=h.label,
            proba={h.label: h.confidence},
            features=h.features,
        )

    try:
        probs_arr = model.predict_proba(feats)[0]
        classes = getattr(model, "classes_", list(range(len(probs_arr))))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"regime.predictor_ml: model prediction failed, fallback: {e}")
        h = heuristic_detect(features_path_s3)
        return RegimeProbs(
            label=h.label,
            proba={h.label: h.confidence},
            features=h.features,
        )

    label_names = [
        "trend_up",
        "trend_down",
        "range",
        "volatility",
        "healthy_rise",
        "crash",
    ]
    proba = {
        label_names[int(classes[i])]: float(probs_arr[i])
        for i in range(len(probs_arr))
    }
    label = max(proba.items(), key=lambda x: x[1])[0]
    return RegimeProbs(label=label, proba=proba, features=feat_dict)
