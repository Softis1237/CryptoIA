from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from loguru import logger

from ..infra.s3 import download_bytes
from .regime_detect import detect as heuristic_detect


@dataclass
class RegimeProbs:
    label: str
    proba: Dict[str, float]
    features: Dict[str, float]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def predict(features_path_s3: str) -> RegimeProbs:
    """ML-based regime classifier placeholder.

    If REGIME_MODEL_PATH / REGIME_MODEL_S3 is configured and loadable, it should
    be used here. As a placeholder, convert heuristic features into a simple
    probability distribution.
    """
    try:
        raw = download_bytes(features_path_s3)
        table = pq.read_table(pa.BufferReader(raw))
        df = table.to_pandas().sort_values("ts").reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"regime.predictor_ml: failed to read features, fallback to heuristic: {e}")
        h = heuristic_detect(features_path_s3)
        return RegimeProbs(label=h.label, proba={h.label: h.confidence, **{k: 0.0 for k in ["trend_up", "trend_down", "range"] if k != h.label}}, features=h.features)

    # Extract proxy features for placeholder
    close = df["close"].astype(float)
    ema20 = df.get("ema_20", close.ewm(span=20, adjust=False).mean()).astype(float)
    ema50 = df.get("ema_50", close.ewm(span=50, adjust=False).mean()).astype(float)
    atr = df.get("atr_14").astype(float) if "atr_14" in df.columns else (df["high"] - df["low"]).rolling(14).mean().fillna(method="bfill").astype(float)

    trend = float(((ema20 - ema50) / (close.abs() + 1e-9)).ewm(span=20, adjust=False).mean().iloc[-1])
    vol = float((atr.iloc[-1] / max(1e-9, close.iloc[-1])) * 100.0)

    # Additional heuristics for crash/healthy rise
    try:
        ret_30 = float(close.pct_change(30).fillna(0.0).iloc[-1] * 100.0)
    except Exception:
        ret_30 = 0.0

    # Map to logits (toy calibration with 6 classes)
    logit_up = trend * 800 - vol * 0.5
    logit_down = -trend * 800 - vol * 0.5
    logit_range = -(abs(trend) * 600) - vol * 0.2
    logit_vol = vol * 0.8 - abs(trend) * 200
    # Healthy rise: positive trend, moderate vol
    vol_penalty = -((max(0.0, vol - 2.0)) ** 2) * 0.1
    logit_healthy = trend * 900 + vol_penalty
    # Crash: sharp negative short-term return and high vol
    logit_crash = max(0.0, -ret_30) * 2.5 + max(0.0, vol - 2.0) * 0.5 - max(0.0, trend) * 300

    logits = np.array([
        logit_up,
        logit_down,
        logit_range,
        logit_vol,
        logit_healthy,
        logit_crash,
    ], dtype=float)
    probs = _softmax(logits)
    labels = ["trend_up", "trend_down", "range", "volatility", "healthy_rise", "crash"]
    proba = {labels[i]: float(probs[i]) for i in range(len(labels))}
    label = max(proba.items(), key=lambda x: x[1])[0]
    return RegimeProbs(label=label, proba=proba, features={"trend": trend, "vol_pct": vol, "ret_30m_pct": ret_30})
