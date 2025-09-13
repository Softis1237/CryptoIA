from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import math

from loguru import logger

from ..infra.db import fetch_news_facts_by_type


@dataclass
class EventStudyInput:
    event_type: str  # e.g., HACK, ETF_APPROVAL, SEC_ACTION
    k: int = 10      # how many past events to include
    window_hours: int = 24  # +/- hours around event to measure
    symbol: str = "BTC/USDT"
    provider: str = "binance"


def _pct_change_around(ts_iso: str, window_h: int, symbol: str, provider: str) -> Optional[float]:
    try:
        import ccxt  # type: ignore
        import pandas as pd

        ex = getattr(ccxt, provider)({"enableRateLimit": True})
        t0 = pd.Timestamp(ts_iso, tz="UTC").to_pydatetime()
        # Fetch +- window_h hours around event using 1m to be precise
        start = int((t0.timestamp() - window_h * 3600) * 1000)
        limit = min(2000, window_h * 120 + 10)
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1m", since=start, limit=limit)
        if not ohlcv:
            return None
        # Find nearest candle to event time and window end
        tgt = int(t0.timestamp() * 1000)
        row0 = min(ohlcv, key=lambda r: abs(r[0] - tgt))
        # after window
        end_ts = int((t0.timestamp() + window_h * 3600) * 1000)
        row1 = min(ohlcv, key=lambda r: abs(r[0] - end_ts))
        p0 = float(row0[4])
        p1 = float(row1[4])
        if p0 <= 0:
            return None
        return (p1 - p0) / p0
    except Exception:
        return None


def run(inp: EventStudyInput) -> Dict[str, Any]:
    """Study average price reaction to past events of the same type.

    Returns summary dict with basic stats and per-sample snippets.
    """
    facts = fetch_news_facts_by_type(inp.event_type, limit=max(1, int(inp.k * 3)))
    # Sort by confidence * magnitude desc to prioritize stronger events
    def _score(row) -> float:
        _, _dir, mag, conf, _ = row
        try:
            return float((mag or 0.0) * (conf or 0.0))
        except Exception:
            return 0.0

    facts = sorted(facts, key=_score, reverse=True)[: max(1, int(inp.k))]
    changes: List[float] = []
    samples: List[Dict[str, Any]] = []
    for ts_iso, direction, mag, conf, src_id in facts:
        ch = _pct_change_around(ts_iso, inp.window_hours, inp.symbol, inp.provider)
        if ch is None:
            continue
        changes.append(float(ch))
        samples.append({
            "ts": ts_iso,
            "direction": direction,
            "magnitude": mag,
            "confidence": conf,
            "pct_change": ch,
        })

    if not changes:
        return {"status": "no-samples", "event_type": inp.event_type}

    avg = sum(changes) / len(changes)
    # fraction of cases where |change| > 3%
    big = sum(1 for x in changes if abs(x) > 0.03)
    pos = sum(1 for x in changes if x > 0)
    neg = sum(1 for x in changes if x < 0)
    std = 0.0
    try:
        m = avg
        std = math.sqrt(sum((x - m) ** 2 for x in changes) / max(1, len(changes) - 1))
    except Exception:
        std = 0.0

    return {
        "event_type": inp.event_type,
        "window_hours": inp.window_hours,
        "n": len(changes),
        "avg_change": avg,
        "std_change": std,
        "p_positive": pos / len(changes),
        "p_negative": neg / len(changes),
        "p_big_move": big / len(changes),
        "samples": samples[:10],
    }

