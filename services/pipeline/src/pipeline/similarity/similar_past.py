from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from ..infra.s3 import download_bytes
from ..infra.db import upsert_feature_embedding, query_similar


@dataclass
class SimilarPastInput:
    features_path_s3: str
    symbol: str = "BTCUSDT"
    k: int = 5


@dataclass
class Neighbor:
    period: str
    distance: float
    outcome_4h: float | None = None
    outcome_12h: float | None = None


def _embed_from_df(df: pd.DataFrame) -> List[float]:
    # Build a compact 16-dim embedding from latest window
    x = df.sort_values("ts").reset_index(drop=True)
    # recent window sizes
    win = [15, 30, 60, 120]
    close = x["close"].astype(float)
    ema20 = x.get("ema_20", close.ewm(span=20, adjust=False).mean()).astype(float)
    ema50 = x.get("ema_50", close.ewm(span=50, adjust=False).mean()).astype(float)
    rsi = x.get("rsi_14")
    if rsi is None:
        # quick compute
        delta = close.diff().fillna(0)
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
    atr = x.get("atr_14")
    if atr is None:
        high = x["high"].astype(float)
        low = x["low"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    feats = []
    for w in win:
        segment = close.iloc[-w:]
        feats.extend([
            float(segment.pct_change().dropna().mean()),
            float(segment.pct_change().dropna().std() or 0.0),
            float((ema20.iloc[-1] - ema50.iloc[-1]) / (close.iloc[-1] + 1e-9)),
            float((rsi.iloc[-w:].mean() - 50.0) / 50.0),
            float((atr.iloc[-w:].mean() / max(1e-9, close.iloc[-1]))),
        ])
    # Truncate/pad to 16 dims
    vec = feats[:16]
    if len(vec) < 16:
        vec = vec + [0.0] * (16 - len(vec))
    # Normalize
    arr = np.array(vec, dtype=float)
    denom = np.linalg.norm(arr) or 1.0
    arr = arr / denom
    return arr.astype(float).tolist()


def run(inp: SimilarPastInput) -> List[Neighbor]:
    raw = download_bytes(inp.features_path_s3)
    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas()
    emb = _embed_from_df(df)

    # Store current embedding by last timestamp with meta (trend/vol proxy)
    ts_window = pd.to_datetime(df["ts"].max(), unit="s", utc=True).isoformat()
    try:
        close = df["close"].astype(float)
        ema20 = df.get("ema_20", close.ewm(span=20, adjust=False).mean()).astype(float)
        ema50 = df.get("ema_50", close.ewm(span=50, adjust=False).mean()).astype(float)
        atr = df.get("atr_14").astype(float)
        trend = float(((ema20 - ema50) / (close.abs() + 1e-9)).ewm(span=20, adjust=False).mean().iloc[-1])
        vol = float((atr.iloc[-1] / max(1e-9, close.iloc[-1])) * 100.0)
        meta = {"source": "features_calc", "trend": trend, "vol_pct": vol}
    except Exception:
        meta = {"source": "features_calc"}
    upsert_feature_embedding(ts_window, inp.symbol, emb, meta)

    rows = query_similar(emb, k=inp.k)
    neighbors: List[Neighbor] = [Neighbor(period=ts, distance=dist) for ts, dist in rows]
    logger.info(f"similar_past: found {len(neighbors)} neighbors")
    return neighbors
