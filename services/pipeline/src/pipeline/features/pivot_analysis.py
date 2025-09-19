from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class PivotPoint:
    idx: int
    ts: pd.Timestamp | None
    price: float
    kind: str  # "high" | "low"


def find_pivots(ohlcv: pd.DataFrame, left_bars: int = 3, right_bars: int = 3) -> pd.DataFrame:
    """Найти опорные точки (локальные экстремумы) по OHLCV.

    Определение:
      - pivot high: high[i] выше high на left_bars слева и right_bars справа
      - pivot low:  low[i] ниже low на left_bars слева и right_bars справа

    Возвращает DataFrame с колонками: idx (int), ts (Timestamp|None), price (float), kind ("high"|"low").
    Индексация соответствует исходным строкам ohlcv (0..n-1).
    """
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(columns=["idx", "ts", "price", "kind"])  # empty
    h = ohlcv["high"].astype(float)
    l = ohlcv["low"].astype(float)
    n = len(ohlcv)
    res: List[PivotPoint] = []

    for i in range(left_bars, n - right_bars):
        # local high
        left_high_ok = (h.iloc[i] > h.iloc[i - left_bars : i]).all() if left_bars > 0 else True
        right_high_ok = (h.iloc[i] > h.iloc[i + 1 : i + 1 + right_bars]).all() if right_bars > 0 else True
        if left_high_ok and right_high_ok:
            ts = _extract_ts(ohlcv, i)
            res.append(PivotPoint(idx=i, ts=ts, price=float(h.iloc[i]), kind="high"))
            continue
        # local low
        left_low_ok = (l.iloc[i] < l.iloc[i - left_bars : i]).all() if left_bars > 0 else True
        right_low_ok = (l.iloc[i] < l.iloc[i + 1 : i + 1 + right_bars]).all() if right_bars > 0 else True
        if left_low_ok and right_low_ok:
            ts = _extract_ts(ohlcv, i)
            res.append(PivotPoint(idx=i, ts=ts, price=float(l.iloc[i]), kind="low"))

    out = pd.DataFrame([p.__dict__ for p in res], columns=["idx", "ts", "price", "kind"]) if res else pd.DataFrame(columns=["idx", "ts", "price", "kind"])  # noqa: E501
    return out


def _extract_ts(df: pd.DataFrame, i: int) -> pd.Timestamp | None:
    # try typical columns first
    for col in ("ts", "timestamp", "time", "dt"):
        if col in df.columns:
            try:
                return pd.to_datetime(df.iloc[i][col], utc=True)
            except Exception:
                break
    # else if index is datetime
    try:
        idx_val = df.index[i]
        if isinstance(idx_val, pd.Timestamp):
            return idx_val
    except Exception:
        pass
    return None

