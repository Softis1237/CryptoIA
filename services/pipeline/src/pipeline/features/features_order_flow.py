from __future__ import annotations

from typing import Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..infra.s3 import download_bytes


def _load_trades(trades_s3: str) -> pd.DataFrame:
    raw = download_bytes(trades_s3)
    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas().sort_values("ts").reset_index(drop=True)
    # ensure required columns
    for c in ["ts", "price", "amount", "side"]:
        if c not in df.columns:
            df[c] = 0 if c != "side" else ""
    return df


def aggregate_metrics(trades_s3: str, freq: str = "1min") -> pd.DataFrame:
    """Aggregate order flow trades into per-minute metrics.

    Output columns: ts (int seconds), ofi, delta_vol, trades_per_sec, avg_trade_size.
    """
    df = _load_trades(trades_s3)
    if df.empty:
        return pd.DataFrame(columns=["ts", "ofi", "delta_vol", "trades_per_sec", "avg_trade_size"])  # type: ignore
    # Build datetime index
    dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index(dt)
    buy = df[df["side"] == "buy"]["amount"].resample(freq).sum().fillna(0.0)
    sell = df[df["side"] == "sell"]["amount"].resample(freq).sum().fillna(0.0)
    total = df["amount"].resample(freq).sum().fillna(0.0)
    count = df["amount"].resample(freq).count().fillna(0)
    ofi = (buy - sell) / total.replace(0.0, pd.NA)
    ofi = ofi.fillna(0.0)
    delta_vol = (buy - sell).fillna(0.0)
    trades_per_sec = (count / 60.0).astype(float)
    avg_trade_size = (total / count.replace(0, pd.NA)).fillna(0.0)
    out = pd.DataFrame(
        {
            "ofi": ofi,
            "delta_vol": delta_vol,
            "trades_per_sec": trades_per_sec,
            "avg_trade_size": avg_trade_size,
        }
    ).reset_index()
    out["ts"] = (out["index"].astype("int64") // 10**9).astype(int)
    out = out.drop(columns=["index"])  # type: ignore
    return out


def latest_metrics(trades_s3: str) -> Dict[str, float]:
    """Compute metrics on the latest aggregation window for feature enrichment."""
    agg = aggregate_metrics(trades_s3)
    if agg.empty:
        return {"ofi_1m": 0.0, "delta_vol_1m": 0.0, "trades_per_sec": 0.0, "avg_trade_size": 0.0}
    last = agg.tail(1).iloc[0]
    return {
        "ofi_1m": float(last.get("ofi", 0.0) or 0.0),
        "delta_vol_1m": float(last.get("delta_vol", 0.0) or 0.0),
        "trades_per_sec": float(last.get("trades_per_sec", 0.0) or 0.0),
        "avg_trade_size": float(last.get("avg_trade_size", 0.0) or 0.0),
    }

