from __future__ import annotations

"""
LivePortfolioManager — live‑учёт портфеля через ccxt или paper_trading (БД).

Назначение:
  - получить открытые позиции, нереализованный PnL, equity/баланс;
  - кэшировать ответы на короткое время для уменьшения нагрузок;
  - дать унифицированный метод get_position(symbol) как у PortfolioManager.

Env:
  - ENABLE_FUTURES=0/1 — использовать фьючерсные позиции (fetch_positions);
  - CCXT_PROVIDER=binance|bybit|okx|...;
  - LIVE_PM_CACHE_SEC=5 — TTL кэша;
  - LIVE_PM_USE_PAPER=0/1 — вместо ccxt читать состояние из таблиц paper_* (для отладки).
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Position:
    symbol: str
    direction: str  # long|short
    quantity: float
    avg_price: float
    unrealized_pnl: float


class LivePortfolioManager:
    def __init__(self) -> None:
        self.provider = os.getenv("CCXT_PROVIDER", "binance")
        try:
            self._ttl = float(os.getenv("LIVE_PM_CACHE_SEC", "5"))
        except Exception:
            self._ttl = 5.0
        self._cache: Dict[str, tuple[float, Any]] = {}
        self._futures = os.getenv("ENABLE_FUTURES", "0") in {"1", "true", "True"}
        self._use_paper = os.getenv("LIVE_PM_USE_PAPER", "0") in {"1", "true", "True"}

    # --- public API -----------------------------------------------------
    def get_position(self, symbol: str) -> Optional[Position]:
        if self._use_paper:
            return self._get_position_paper(symbol)
        return self._get_position_ccxt(symbol)

    def total_equity(self) -> float:
        if self._use_paper:
            return self._get_equity_paper()
        bal = self._get_balance_ccxt()
        # best‑effort: USDT equity if present, else sum of total balances
        try:
            usdt = bal.get("total", {}).get("USDT")
            if usdt is not None:
                return float(usdt)
            return float(sum(float(v or 0.0) for v in (bal.get("total") or {}).values()))
        except Exception:
            return 0.0

    def free_balance(self, currency: str = "USDT") -> float:
        if self._use_paper:
            return self._get_equity_paper()
        bal = self._get_balance_ccxt()
        try:
            return float((bal.get("free") or {}).get(currency) or 0.0)
        except Exception:
            return 0.0

    # --- ccxt branch ----------------------------------------------------
    def _get_ex(self):
        import ccxt  # type: ignore

        return getattr(ccxt, self.provider)({"enableRateLimit": True})

    def _cache_get(self, key: str):
        now = time.time()
        v = self._cache.get(key)
        if v and (now - v[0]) <= self._ttl:
            return v[1]
        return None

    def _cache_put(self, key: str, value: Any):
        self._cache[key] = (time.time(), value)

    def _get_balance_ccxt(self) -> Dict[str, Any]:
        cached = self._cache_get("balance")
        if cached is not None:
            return cached
        try:
            ex = self._get_ex()
            data = ex.fetch_balance() or {}
            self._cache_put("balance", data)
            return data
        except Exception:
            return {}

    def _get_ticker(self, symbol: str) -> float:
        key = f"ticker:{symbol}"
        cached = self._cache_get(key)
        if cached is not None:
            return float(cached)
        try:
            ex = self._get_ex()
            t = ex.fetch_ticker(symbol)
            px = float(t.get("last") or t.get("close") or 0.0)
            self._cache_put(key, px)
            return px
        except Exception:
            return 0.0

    def _get_position_ccxt(self, symbol: str) -> Optional[Position]:
        try:
            ex = self._get_ex()
            if self._futures and hasattr(ex, "fetch_positions"):
                key = "positions"
                cached = self._cache_get(key)
                rows = cached
                if rows is None:
                    rows = ex.fetch_positions() or []  # type: ignore[attr-defined]
                    self._cache_put(key, rows)
                for p in rows:
                    sym = str(p.get("symbol") or p.get("info", {}).get("symbol") or "")
                    if sym.replace(":USDT", "/USDT").replace("USDT", "/USDT") != symbol:
                        continue
                    amt = float(p.get("contracts") or p.get("amount") or p.get("positionAmt") or 0.0)
                    side = "long" if amt > 0 else ("short" if amt < 0 else "")
                    if not side:
                        return None
                    qty = abs(amt)
                    avg = float(p.get("entryPrice") or p.get("avgPrice") or 0.0)
                    mark = float(p.get("markPrice") or self._get_ticker(symbol) or avg)
                    upnl = (mark - avg) * qty if side == "long" else (avg - mark) * qty
                    return Position(symbol=symbol, direction=side, quantity=qty, avg_price=avg, unrealized_pnl=upnl)
            # spot: нет понятия открытой позиции — возвращаем None
            return None
        except Exception:
            return None

    # --- paper (DB) branch ---------------------------------------------
    def _get_position_paper(self, symbol: str) -> Optional[Position]:
        from ..infra.db import get_conn

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT side, entry, qty FROM paper_positions WHERE status='OPEN' ORDER BY opened_at DESC LIMIT 1"
                )
                row = cur.fetchone()
                if not row:
                    return None
                side, entry, qty = row
                side = str(side).lower()
                entry = float(entry)
                qty = float(qty)
                # best effort mark via ccxt
                mark = self._get_ticker(symbol)
                upnl = (mark - entry) * qty if side == "long" else (entry - mark) * qty
                return Position(symbol=symbol, direction=side, quantity=qty, avg_price=entry, unrealized_pnl=upnl)

    def _get_equity_paper(self) -> float:
        from ..infra.db import get_conn

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT equity FROM paper_accounts ORDER BY created_at DESC LIMIT 1")
                row = cur.fetchone()
                return float(row[0] or 0.0) if row else 0.0

