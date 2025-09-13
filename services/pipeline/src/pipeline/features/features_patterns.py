from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..infra.db import fetch_technical_patterns


def _safe_abs(x: pd.Series) -> pd.Series:
    try:
        return (x.astype(float)).abs()
    except Exception:
        return x.abs()


def detect_patterns(df: pd.DataFrame, patterns: List[dict] | None = None) -> Dict[str, pd.Series]:
    """Detect a minimal set of classic patterns over the whole series.

    Returns mapping column_name -> 0/1 pd.Series aligned to df index.
    """
    if df.empty:
        return {}
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    body = _safe_abs(c - o)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    eps = 1e-9

    # Hammer / Shooting star
    hammer = ((lower >= 2.0 * body) & (upper <= 0.4 * body)).astype(int)
    shooting = ((upper >= 2.0 * body) & (lower <= 0.4 * body)).astype(int)

    # Engulfing
    o1, c1 = o.shift(1), c.shift(1)
    body1 = _safe_abs(c1 - o1)
    green = (c > o)
    red = (c < o)
    prev_green = (c1 > o1)
    prev_red = (c1 < o1)
    bull_engulf = (
        prev_red & green & (o <= c1) & (c >= o1) & (_safe_abs(c - o) > 1.05 * body1)
    ).astype(int)
    bear_engulf = (
        prev_green & red & (o >= c1) & (c <= o1) & (_safe_abs(c - o) > 1.05 * body1)
    ).astype(int)

    # Morning Star (approximate heuristic)
    o2, c2 = o.shift(2), c.shift(2)
    body2 = _safe_abs(c2 - o2)
    small_mid = _safe_abs(c1 - o1) < (body2 * 0.6)
    c1_red = c2 < o2
    c3_green = green
    close3_above_mid1 = c > ((o2 + c2) / 2.0)
    morning_star = (c1_red & small_mid & c3_green & close3_above_mid1).astype(int)

    # Aggregate score
    score = hammer + shooting + bull_engulf + bear_engulf + morning_star

    return {
        "pat_hammer": hammer,
        "pat_shooting_star": shooting,
        "pat_engulfing_bull": bull_engulf,
        "pat_engulfing_bear": bear_engulf,
        "pat_morning_star": morning_star,
        "pat_score": score,
    }


def load_patterns_from_db() -> List[dict]:
    """Fetch raw pattern definitions (not yet used by rules above)."""
    try:
        return fetch_technical_patterns()
    except Exception:
        return []

