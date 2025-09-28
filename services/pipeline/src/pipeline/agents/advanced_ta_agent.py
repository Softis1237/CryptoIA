from __future__ import annotations

"""Advanced Technical Analysis agent.

Computes Fibonacci retracement/extension levels and summarises the swing
structure for downstream agents.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ..infra import metrics
from ..infra.s3 import download_bytes
from .base import AgentResult, BaseAgent


class AdvancedTAPayload:
    """Lightweight payload validator without pulling extra dependencies."""

    __slots__ = ("features_path_s3", "ohlcv", "symbol", "lookback", "timeframe", "slot")

    def __init__(
        self,
        *,
        features_path_s3: Optional[str] = None,
        ohlcv: Optional[Iterable[dict]] = None,
        symbol: str = "BTC/USDT",
        lookback: int = 120,
        timeframe: str = "4h",
        slot: str = "advanced_ta",
    ) -> None:
        if not features_path_s3 and not ohlcv:
            raise ValueError("features_path_s3 or ohlcv must be provided")
        self.features_path_s3 = features_path_s3
        self.ohlcv = ohlcv
        self.symbol = symbol
        self.lookback = max(20, int(lookback))
        self.timeframe = timeframe
        self.slot = slot


def _load_from_features(s3_path: str) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = download_bytes(s3_path)
    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas().sort_values("ts").reset_index(drop=True)
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("features parquet must contain OHLC columns")
    return df


def _load_from_iterable(rows: Iterable[dict]) -> pd.DataFrame:
    data = list(rows)
    if not data:
        raise ValueError("ohlcv iterable is empty")
    df = pd.DataFrame(data)
    expected = {"open", "high", "low", "close"}
    if not expected.issubset(df.columns):
        raise ValueError(f"ohlcv iterable missing columns {expected - set(df.columns)}")
    if "ts" not in df.columns:
        df["ts"] = range(len(df))
    return df.sort_values("ts").reset_index(drop=True)


def _resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    dt = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    if dt.isna().all():
        # ts already datetime/ordinal
        dt = pd.to_datetime(df["ts"], errors="coerce")
    if dt.isna().all():
        # fallback: treat as index
        df = df.copy()
        df.index = pd.Index(range(len(df)))
        return df
    df = df.copy()
    df.index = dt
    tf = timeframe.lower()
    rule = "4h"
    if tf.endswith("m"):
        rule = f"{int(tf[:-1])}min"
    elif tf.endswith("h"):
        rule = f"{int(tf[:-1])}h"
    elif tf.endswith("d"):
        rule = f"{int(tf[:-1])}d"
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    res = df.resample(rule).agg(agg).dropna()
    return res.reset_index(drop=True)


def _find_swing(df: pd.DataFrame, lookback: int) -> Tuple[float, float]:
    tail = df.tail(lookback)
    swing_high = float(tail["high"].max())
    swing_low = float(tail["low"].min())
    if not math.isfinite(swing_high) or not math.isfinite(swing_low):
        raise ValueError("invalid swing bounds")
    return swing_high, swing_low


def _fib_levels(high: float, low: float) -> Dict[str, List[float]]:
    diff = max(high - low, 1e-9)
    supports = [high - diff * ratio for ratio in (0.382, 0.5, 0.618)]
    resistances = [high + diff * ratio for ratio in (0.618, 1.0, 1.618)]
    extensions = [low - diff * ratio for ratio in (0.382, 0.618)]
    return {
        "fib_support": [round(v, 2) for v in supports],
        "fib_resistance": [round(v, 2) for v in resistances],
        "fib_extension": [round(v, 2) for v in extensions],
    }


def _trend_bias(high: float, low: float, close: float) -> str:
    mid = (high + low) / 2.0
    if close >= high * 0.97:
        return "up_swing"
    if close <= low * 1.03:
        return "down_swing"
    if close >= mid:
        return "range_up"
    return "range_down"


@dataclass(slots=True)
class AdvancedTAAgent(BaseAgent):
    name: str = "advanced-ta-agent"
    priority: int = 32

    def run(self, payload: dict) -> AgentResult:
        params = AdvancedTAPayload(**payload)
        if params.features_path_s3:
            df = _load_from_features(params.features_path_s3)
        else:
            df = _load_from_iterable(params.ohlcv or [])
        timeframe = params.timeframe or "4h"
        df_tf = _resample(df, timeframe)
        lookback = min(len(df_tf), params.lookback)
        if lookback < 20:
            raise ValueError("not enough data for fibonacci analysis")
        swing_high, swing_low = _find_swing(df_tf, lookback)
        close = float(df_tf["close"].iloc[-1])
        fib = _fib_levels(swing_high, swing_low)
        bias = _trend_bias(swing_high, swing_low, close)
        span = swing_high - swing_low
        summary = (
            f"Swing {timeframe}: high={swing_high:.2f}, low={swing_low:.2f}, range={span:.2f}. "
            f"Close={close:.2f}, bias={bias}."
        )
        output = {
            "symbol": params.symbol,
            "timeframe": timeframe,
            "lookback": lookback,
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "range": round(span, 2),
            "close": round(close, 2),
            "bias": bias,
            "summary": summary,
            **fib,
        }
        try:
            metrics.push_values(
                job=self.name,
                values={
                    "range": float(span),
                    "close": float(close),
                },
                labels={"timeframe": timeframe, "bias": bias},
            )
        except Exception:
            pass
        return AgentResult(name=self.name, ok=True, output=output)


def main() -> None:  # pragma: no cover - manual smoke
    import json
    import sys

    payload = {
        "features_path_s3": sys.argv[1] if len(sys.argv) > 1 else None,
        "symbol": "BTC/USDT",
    }
    agent = AdvancedTAAgent()
    res = agent.run(payload)
    print(json.dumps(res.output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
