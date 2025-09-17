from __future__ import annotations

"""
SMC‑зоны (Order Blocks, FVG, Liquidity Pools, Breaker Blocks) на OHLCV.

Методы возвращают список зон формата dict:
  {
    "zone_type": "OB_BULL" | "OB_BEAR" | "FVG" | "LIQUIDITY_POOL" | "BREAKER",
    "price_low": float,
    "price_high": float,
    "status": "untested" | "mitigated" | "invalidated",
    "meta": {...},
  }

Эвристики упрощены, параметры можно калибровать под конкретные таймфреймы.
"""

from typing import Any, Dict, List
import hashlib

import pandas as pd


def _zones_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(str(sorted(payload.items())).encode("utf-8")).hexdigest()


def detect_order_blocks(
    df: pd.DataFrame, kind: str = "auto", lookback: int = 60, timeframe: str | None = None
) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    x = df.tail(max(lookback, 20)).copy()
    # Простой подход: последняя импульсная свеча (тело > 1.5 * медианы) задаёт зону OB
    body = (x["close"].astype(float) - x["open"].astype(float)).abs()
    med = float(body.median() or 0.0)
    if med <= 0:
        return []
    threshold = 1.5 * med
    idx = body[body >= threshold].index
    if len(idx) == 0:
        return []
    i = int(idx[-1])
    row = df.loc[i]
    last = df.iloc[-1]
    is_bull = float(row["close"]) > float(row["open"])
    lo = float(min(row["open"], row["close"]))
    hi = float(max(row["open"], row["close"]))
    # status: mitigated if last bar overlapped the zone; invalidated if broke beyond zone with slack
    last_low = float(last["low"]) if "low" in df.columns else float(last["close"])  # type: ignore[index]
    last_high = float(last["high"]) if "high" in df.columns else float(last["close"])  # type: ignore[index]
    last_close = float(last["close"])  # type: ignore[index]
    touched = (last_high >= lo) and (last_low <= hi)
    # tolerance by timeframe (bps)
    tf = (timeframe or "").lower()
    tol_bps = 30 if tf in {"4h", "1d"} else 20 if tf in {"1h"} else 10
    eps = ((lo + hi) / 2.0) * (tol_bps / 1e4)
    if is_bull:
        invalid = last_close < (lo - eps)
    else:
        invalid = last_close > (hi + eps)
    status = "invalidated" if invalid else ("mitigated" if touched else "untested")
    zone_type = "OB_BULL" if is_bull else "OB_BEAR"
    return [
        {
            "zone_type": zone_type,
            "price_low": lo,
            "price_high": hi,
            "status": status,
            "meta": {"idx": i},
        }
    ]


def detect_fvg(df: pd.DataFrame, lookback: int = 60, timeframe: str | None = None) -> List[Dict[str, Any]]:
    if len(df) < 3:
        return []
    x = df.tail(max(lookback, 20)).copy()
    # FVG (имбаланс): gap между high(бар n-1) и low(бар n+1)
    h1 = x["high"].astype(float).shift(1)
    l1 = x["low"].astype(float).shift(-1)
    gap_up = (l1 - h1) > 0
    gap_down = (h1 - l1) > 0
    zones: List[Dict[str, Any]] = []
    for i, ok in enumerate(gap_up.tail(5).items()):
        pass
    # последняя зона
    last_idx = x.index[-2]  # центр на текущем баре
    hi_prev = float(df["high"].iloc[-2])
    lo_next = float(df["low"].iloc[-1])
    if lo_next > hi_prev:
        zones.append({
            "zone_type": "FVG",
            "price_low": hi_prev,
            "price_high": lo_next,
            "status": "untested",
            "meta": {"idx": int(last_idx)},
        })
    # bearish FVG
    hi_next = float(df["high"].iloc[-1])
    lo_prev = float(df["low"].iloc[-2])
    if lo_prev > hi_next:
        zones.append({
            "zone_type": "FVG",
            "price_low": hi_next,
            "price_high": lo_prev,
            "status": "untested",
            "meta": {"idx": int(last_idx)},
        })
    # set status for latest close
    if zones:
        last_close = float(df["close"].iloc[-1])
        tf = (timeframe or "").lower()
        tol_bps = 30 if tf in {"4h", "1d"} else 20 if tf in {"1h"} else 10
        for z in zones:
            lo = float(z["price_low"])
            hi = float(z["price_high"])
            eps = ((lo + hi) / 2.0) * (tol_bps / 1e4)
            if lo <= last_close <= hi:
                z["status"] = "mitigated"
            elif last_close > (hi + eps) or last_close < (lo - eps):
                z["status"] = "invalidated"
    return zones


def detect_liquidity_pools(df: pd.DataFrame, window: int = 100) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    x = df.tail(max(30, window)).copy()
    eq_high = float(x["high"].max())
    eq_low = float(x["low"].min())
    return [
        {
            "zone_type": "LIQUIDITY_POOL",
            "price_low": eq_high,
            "price_high": eq_high,
            "status": "untested",
            "meta": {"pool": "high"},
        },
        {
            "zone_type": "LIQUIDITY_POOL",
            "price_low": eq_low,
            "price_high": eq_low,
            "status": "untested",
            "meta": {"pool": "low"},
        },
    ]


def detect_breaker_blocks(df: pd.DataFrame, lookback: int = 80, timeframe: str | None = None) -> List[Dict[str, Any]]:
    # Упрощённо: пробитый OB превращается в breaker. Здесь — последняя крупная свеча с ретестом тела.
    if df.empty:
        return []
    ob = detect_order_blocks(df, lookback=lookback)
    if not ob:
        return []
    z = ob[-1]
    lo, hi = float(z["price_low"]), float(z["price_high"])
    close = float(df["close"].iloc[-1])
    # tolerance by timeframe
    tf = (timeframe or "").lower()
    tol_bps = 30 if tf in {"4h", "1d"} else 20 if tf in {"1h"} else 10
    eps = ((lo + hi) / 2.0) * (tol_bps / 1e4)
    status = "mitigated" if (lo <= close <= hi) else ("invalidated" if (close > hi + eps or close < lo - eps) else "untested")
    return [
        {
            "zone_type": "BREAKER",
            "price_low": lo,
            "price_high": hi,
            "status": status,
            "meta": {"from": z.get("zone_type")},
        }
    ]


def save_zones_to_db(symbol: str, timeframe: str, zones: List[Dict[str, Any]]) -> int:
    try:
        from ..infra.db import insert_smc_zone
    except Exception:
        return 0
    count = 0
    for z in zones:
        payload = dict(z)
        meta = dict(payload.get("meta") or {})
        meta["hash"] = _zones_hash({k: v for k, v in payload.items() if k != "meta"})
        try:
            insert_smc_zone(
                symbol=symbol,
                timeframe=timeframe,
                zone_type=str(payload.get("zone_type")),
                price_low=float(payload.get("price_low") if payload.get("price_low") is not None else 0.0),
                price_high=float(payload.get("price_high") if payload.get("price_high") is not None else 0.0),
                status=str(payload.get("status") or "untested"),
                meta=meta,
            )
            count += 1
        except Exception:
            continue
    return count
