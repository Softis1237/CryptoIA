from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math
import os
import time
import pandas as pd

from ..infra.s3 import upload_bytes
from ..infra.db import upsert_strategic_verdict


_OHLC_CACHE: Dict[tuple[str, str, str], tuple[float, List[List[float]]]] = {}


def _cache_ttl() -> float:
    try:
        return float(os.getenv("SMC_OHLC_CACHE_SEC", "120"))
    except Exception:
        return 120.0


def _reset_cache() -> None:
    _OHLC_CACHE.clear()


@dataclass
class SMCInput:
    run_id: str
    slot: str = "smc"
    symbol: str = "BTC/USDT"
    provider: str = "binance"


def _fetch_ohlcv(symbol: str, provider: str, timeframe: str, limit: int = 500) -> List[List[float]]:
    key = (symbol, provider, timeframe)
    now = time.time()
    cached = _OHLC_CACHE.get(key)
    ttl = _cache_ttl()
    if cached and (now - cached[0]) <= ttl:
        return cached[1]

    import ccxt  # type: ignore

    ex = getattr(ccxt, provider)({"enableRateLimit": True, "timeout": 20000})
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=min(1000, max(50, limit))) or []
    _OHLC_CACHE[key] = (now, data)
    return data


def _hh_hl(df: pd.DataFrame, lookback: int = 30) -> str:
    x = df.tail(lookback)
    highs_up = x["high"].diff().fillna(0.0) > 0
    lows_up = x["low"].diff().fillna(0.0) > 0
    up = highs_up.rolling(5).sum().iloc[-1] > 2 and lows_up.rolling(5).sum().iloc[-1] > 2
    if up:
        return "trend_up"
    highs_dn = x["high"].diff().fillna(0.0) < 0
    lows_dn = x["low"].diff().fillna(0.0) < 0
    dn = highs_dn.rolling(5).sum().iloc[-1] > 2 and lows_dn.rolling(5).sum().iloc[-1] > 2
    return "trend_down" if dn else "range"


def _sweep(df15: pd.DataFrame, htf_levels: Dict[str, float], tol_bps: float = 10.0) -> Optional[str]:
    tol = tol_bps / 1e4
    hi = float(htf_levels.get("eq_high", 0.0) or 0.0)
    lo = float(htf_levels.get("eq_low", 0.0) or 0.0)
    if hi > 0 and (df15["high"].iloc[-1] > hi * (1.0 + tol)):
        return "sweep_high"
    if lo > 0 and (df15["low"].iloc[-1] < lo * (1.0 - tol)):
        return "sweep_low"
    return None


def _first_choch(df5: pd.DataFrame) -> Optional[str]:
    # naive: detect first HH->LL or LL->HH around recent bars
    x = df5.tail(30)
    close = x["close"].astype(float)
    # change of character proxy
    up = close.diff().fillna(0.0).rolling(3).sum().iloc[-1] > 0 and close.iloc[-1] > close.iloc[-5]
    dn = close.diff().fillna(0.0).rolling(3).sum().iloc[-1] < 0 and close.iloc[-1] < close.iloc[-5]
    if up:
        return "bullish"
    if dn:
        return "bearish"
    return None


def _ob_fvg(df5: pd.DataFrame) -> Optional[List[float]]:
    # very coarse: last impulse candle body as OB zone
    x = df5.tail(10)
    body = (x["close"] - x["open"]).abs()
    i = int(body.idxmax())
    row = df5.loc[i]
    lo = float(min(row["open"], row["close"]))
    hi = float(max(row["open"], row["close"]))
    if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
        return [lo, hi]
    return None


def run(inp: SMCInput) -> Dict[str, Any]:
    # 4h context
    h4 = _fetch_ohlcv(inp.symbol, inp.provider, "4h", limit=200)
    if not h4:
        return {"status": "no-data"}
    df4 = pd.DataFrame(h4, columns=["ts", "open", "high", "low", "close", "volume"]).astype(float)
    trend = _hh_hl(df4)
    # equal highs/lows as rough liquidity pools
    eq_high = float(df4["high"].tail(50).max())
    eq_low = float(df4["low"].tail(50).min())
    # 15m sweep
    m15 = _fetch_ohlcv(inp.symbol, inp.provider, "15m", limit=200)
    df15 = pd.DataFrame(m15, columns=["ts", "open", "high", "low", "close", "volume"]).astype(float)
    sweep = _sweep(df15, {"eq_high": eq_high, "eq_low": eq_low})
    # 5m CHOCH + POI
    m5 = _fetch_ohlcv(inp.symbol, inp.provider, "5m", limit=200)
    df5 = pd.DataFrame(m5, columns=["ts", "open", "high", "low", "close", "volume"]).astype(float)
    choch = _first_choch(df5)
    poi = _ob_fvg(df5)
    # Premium/Discount check relative to local range on 5m
    rng_lo = float(df5["low"].tail(100).min())
    rng_hi = float(df5["high"].tail(100).max())
    mid = (rng_lo + rng_hi) / 2.0 if (rng_hi > rng_lo) else 0.0
    zone_ok = None
    if poi and mid > 0:
        # long -> discount (below mid), short -> premium (above mid)
        z_mid = (poi[0] + poi[1]) / 2.0
        if choch == "bullish":
            zone_ok = (z_mid <= mid)
        elif choch == "bearish":
            zone_ok = (z_mid >= mid)

    status = None
    entry_zone: Optional[List[float]] = None
    invalidation: Optional[float] = None
    target: Optional[float] = None
    if trend in {"trend_up", "range"} and sweep in {"sweep_low"} and choch == "bullish" and poi and zone_ok:
        status = "SMC_BULLISH_SETUP"
        entry_zone = poi
        invalidation = min(poi) - (rng_hi - rng_lo) * 0.05
        target = eq_high
    elif trend in {"trend_down", "range"} and sweep in {"sweep_high"} and choch == "bearish" and poi and zone_ok:
        status = "SMC_BEARISH_SETUP"
        entry_zone = poi
        invalidation = max(poi) + (rng_hi - rng_lo) * 0.05
        target = eq_low

    verdict = {
        "status": status or "NO_SETUP",
        "trend_4h": trend,
        "htf_liquidity": {"eq_high": eq_high, "eq_low": eq_low},
        "sweep_15m": sweep,
        "choch_5m": choch,
        "entry_zone": entry_zone,
        "invalidation_level": invalidation,
        "target_liquidity": target,
    }

    # Save to S3 for consumption by reactor (DB integration can be added via migrations)
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    date_key = now.strftime("%Y-%m-%d")
    path = f"runs/{date_key}/{inp.slot}/smc_verdict.json"
    upload_bytes(path, json.dumps(verdict).encode("utf-8"), content_type="application/json")
    try:
        upsert_strategic_verdict(
            agent_name="SMC Analyst",
            symbol=inp.symbol,
            ts=now,
            verdict=str(verdict.get("status")),
            confidence=None,
            meta=verdict,
        )
    except Exception:
        pass
    return verdict


def main() -> None:
    import sys, json
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.agents.smc_analyst '<json_payload>'", file=sys.stderr)
        sys.exit(2)
    payload = SMCInput(**json.loads(sys.argv[1]))
    out = run(payload)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
