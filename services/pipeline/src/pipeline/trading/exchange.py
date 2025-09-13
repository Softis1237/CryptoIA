from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import ccxt
from loguru import logger


def _build_exchange() -> ccxt.Exchange:
    name = os.getenv("EXCHANGE", os.getenv("CCXT_EXCHANGE", os.getenv("CCXT_PROVIDER", "binance")))
    ex_cls = getattr(ccxt, name)
    api_key = os.getenv("EXCHANGE_API_KEY")
    secret = os.getenv("EXCHANGE_SECRET")
    password = os.getenv("EXCHANGE_PASSWORD")
    default_type = os.getenv("EXCHANGE_TYPE", "spot")  # spot|future|swap
    timeout_ms = int(os.getenv("CCXT_TIMEOUT_MS", "20000"))
    params: Dict[str, Any] = {"enableRateLimit": True, "timeout": timeout_ms, "options": {"defaultType": default_type}}
    if api_key and secret:
        params.update({"apiKey": api_key, "secret": secret})
        if password:
            params.update({"password": password})
    ex: ccxt.Exchange = ex_cls(params)
    return ex


@dataclass
class OrderResult:
    id: str
    type: str
    side: str
    amount: float
    price: Optional[float]
    info: Dict[str, Any]


class ExchangeClient:
    def __init__(self) -> None:
        self.ex = _build_exchange()

    def get_last_price(self, symbol: str) -> float:
        t = self.ex.fetch_ticker(symbol)
        px = t.get("last") or t.get("close")
        return float(px)

    def get_balance(self, code: str) -> float:
        bal = self.ex.fetch_balance()
        total = (bal.get("total") or {}).get(code)
        if total is None:
            # try free
            total = (bal.get("free") or {}).get(code, 0.0)
        return float(total or 0.0)

    def create_market(self, symbol: str, side: str, amount: float, params: Optional[dict] = None) -> OrderResult:
        o = self.ex.create_order(symbol, "market", side, amount, None, params or {})
        return OrderResult(id=str(o.get("id")), type="market", side=side, amount=float(amount), price=None, info=o)

    def create_limit(self, symbol: str, side: str, amount: float, price: float, params: Optional[dict] = None) -> OrderResult:
        o = self.ex.create_order(symbol, "limit", side, amount, price, params or {})
        return OrderResult(id=str(o.get("id")), type="limit", side=side, amount=float(amount), price=float(price), info=o)

    def create_stop(self, symbol: str, side: str, amount: float, stop_price: float, reduce_only: bool = True, params: Optional[dict] = None) -> OrderResult:
        p = params.copy() if params else {}
        # Try common keys for conditional orders across exchanges
        p.setdefault("stopPrice", stop_price)
        p.setdefault("triggerPrice", stop_price)
        p.setdefault("reduceOnly", reduce_only)
        # Some exchanges require specific type (e.g., STOP_MARKET)
        order_type = p.pop("_type", None) or "STOP_MARKET"
        try:
            o = self.ex.create_order(symbol, order_type, side, amount, None, p)
        except Exception as e:
            logger.debug(f"create_stop fallback: {e}; trying params-only trigger")
            # fallback to market with params
            o = self.ex.create_order(symbol, "market", side, amount, None, p)
        return OrderResult(id=str(o.get("id")), type="stop", side=side, amount=float(amount), price=None, info=o)

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> None:
        try:
            self.ex.cancel_order(order_id, symbol)
        except Exception:
            pass

    def fetch_open_orders(self, symbol: Optional[str] = None) -> list[dict]:
        try:
            return self.ex.fetch_open_orders(symbol)
        except Exception:
            return []

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.ex.set_leverage(leverage, symbol)
        except Exception:
            pass
