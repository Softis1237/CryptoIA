from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math
import os
import time
import pandas as pd

from ..infra.s3 import upload_bytes
from ..infra.db import fetch_smc_zones, fetch_latest_strategic_verdict, fetch_latest_regime_label
from ..features.features_smc import (
    detect_order_blocks,
    detect_fvg,
    detect_liquidity_pools,
    detect_breaker_blocks,
    save_zones_to_db,
)
from ..infra.db import upsert_strategic_verdict
from .memory_guardian_agent import MemoryGuardianAgent


_OHLC_CACHE: Dict[tuple[str, str, str], tuple[float, List[List[float]]]] = {}
_MEMORY_GUARDIAN = MemoryGuardianAgent()


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
    # v2 mode: rely on structured zones + whale + regime, avoid raw OHLC if possible
    if os.getenv("SMC_V2", "0") in {"1", "true", "True"}:
        regime = fetch_latest_regime_label() or "range"
        zones_15m = fetch_smc_zones(inp.symbol, "15m", status="untested", limit=50)
        zones_1h = fetch_smc_zones(inp.symbol, "1h", status="untested", limit=50)
        whale = fetch_latest_strategic_verdict("Whale Watcher", inp.symbol) or {}
        whale_status = str(whale.get("verdict") or "")
        # Simple rule: HTF up + LTF OB_BULL present + Whale bullish -> bullish setup
        has_bull_ob = any((z.get("zone_type") == "OB_BULL" for z in zones_15m)) or any((z.get("zone_type") == "OB_BULL" for z in zones_1h))
        has_bear_ob = any((z.get("zone_type") == "OB_BEAR" for z in zones_15m)) or any((z.get("zone_type") == "OB_BEAR" for z in zones_1h))
        status = "NO_SETUP"
        if regime == "trend_up" and has_bull_ob and whale_status.endswith("BULLISH"):
            status = "SMC_BULLISH_SETUP"
        elif regime == "trend_down" and has_bear_ob and whale_status.endswith("BEARISH"):
            status = "SMC_BEARISH_SETUP"
        verdict = {
            "status": status,
            "trend_4h": regime,
            "zones": {"1h": zones_1h[-5:], "15m": zones_15m[-5:]},
            "whale": whale,
        }
        memory_alerts = _memory_lookup(inp.symbol, status, regime)
        if memory_alerts:
            verdict["memory_alerts"] = memory_alerts
            negative = [ls for ls in memory_alerts if ls.get("outcome_after", 0.0) < 0]
            if negative and status.startswith("SMC_"):
                verdict["status"] = f"{status}_CAUTION"
        # Persist as before
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

    # 4h context (v1 path)
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

    # optional SMC zones (save if flag set)
    zones_1h = []
    zones_15m = []
    try:
        if os.getenv("SMC_SAVE_ZONES", "1") in {"1", "true", "True"}:
            df4h = df4.copy()
            # простая проекция на 1h/15m: используем 4h/15m/5m как суррогатные источники
            zones_1h = (
                detect_order_blocks(df4h, timeframe="1h")
                + detect_fvg(df4h, timeframe="1h")
                + detect_liquidity_pools(df4h)
                + detect_breaker_blocks(df4h, timeframe="1h")
            )
            zones_15m = (
                detect_order_blocks(df15, timeframe="15m")
                + detect_fvg(df15, timeframe="15m")
                + detect_liquidity_pools(df15)
                + detect_breaker_blocks(df15, timeframe="15m")
            )
            save_zones_to_db(inp.symbol, "1h", zones_1h)
            save_zones_to_db(inp.symbol, "15m", zones_15m)
    except Exception:
        zones_1h, zones_15m = [], []

    verdict = {
        "status": status or "NO_SETUP",
        "trend_4h": trend,
        "htf_liquidity": {"eq_high": eq_high, "eq_low": eq_low},
        "sweep_15m": sweep,
        "choch_5m": choch,
        "entry_zone": entry_zone,
        "invalidation_level": invalidation,
        "target_liquidity": target,
        "zones": {"1h": zones_1h[-3:], "15m": zones_15m[-3:]},
    }
    memory_alerts = _memory_lookup(inp.symbol, verdict["status"], trend)
    if memory_alerts:
        verdict["memory_alerts"] = memory_alerts
        negative = [ls for ls in memory_alerts if ls.get("outcome_after", 0.0) < 0]
        if negative and verdict["status"].startswith("SMC_"):
            verdict["status"] = f"{verdict['status']}_CAUTION"

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


def _memory_lookup(symbol: str, setup: str, regime: str) -> List[dict]:
    if not setup or setup == "NO_SETUP":
        return []
    try:
        res = _MEMORY_GUARDIAN.query(
            {
                "scope": "trading",
                "context": {
                    "symbol": symbol,
                    "planned_signal": setup,
                    "market_regime": regime,
                },
                "top_k": 3,
            }
        )
        return res.output.get("lessons", []) if res and isinstance(res.output, dict) else []
    except Exception:
        return []


if __name__ == "__main__":
    main()
