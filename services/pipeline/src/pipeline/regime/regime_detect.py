from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..infra.s3 import download_bytes


@dataclass
class Regime:
    label: str
    confidence: float
    features: dict


def _trend_strength(close: pd.Series, ema_fast: pd.Series, ema_slow: pd.Series) -> float:
    # Normalized distance between EMAs, signed by direction
    eps = 1e-9
    dist = (ema_fast - ema_slow) / (close.abs() + eps)
    return float(dist.ewm(span=20, adjust=False).mean().iloc[-1])


def _volatility_pct(atr: pd.Series, close: pd.Series) -> float:
    return float((atr.iloc[-1] / max(1e-9, close.iloc[-1])) * 100.0)


def detect(features_path_s3: str) -> Regime:
    raw = download_bytes(features_path_s3)
    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas().sort_values("ts").reset_index(drop=True)
    close = df["close"].astype(float)
    ema_fast = df.get("ema_20", close.ewm(span=20, adjust=False).mean()).astype(float)
    ema_slow = df.get("ema_50", close.ewm(span=50, adjust=False).mean()).astype(float)
    atr = df.get("atr_14")
    if atr is None:
        # simple proxy
        atr = (df["high"] - df["low"]).rolling(14).mean().fillna(method="bfill")
    atr = atr.astype(float)

    trend = _trend_strength(close, ema_fast, ema_slow)
    vol_pct = _volatility_pct(atr, close)

    if trend > 0.0015:
        label = "trend_up"
    elif trend < -0.0015:
        label = "trend_down"
    else:
        label = "range"

    # Confidence rises with |trend| and falls with volatility
    conf = float(min(1.0, max(0.0, abs(trend) * 800 - (vol_pct / 200))))

    return Regime(label=label, confidence=round(conf, 2), features={"trend": trend, "vol_pct": vol_pct})
