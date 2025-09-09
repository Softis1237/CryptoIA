from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from ..infra.db import get_conn
from .engine import TradingEngine


def _now() -> datetime:
    return datetime.now(timezone.utc)


def open_from_last_suggestion(symbol: str = os.getenv("EXEC_SYMBOL", "BTC/USDT")) -> Optional[str]:
    if os.getenv("EXECUTE_LIVE", "0") not in {"1", "true", "True"}:
        logger.info("EXECUTE_LIVE=0: skip live execution")
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT run_id, side, entry_zone, leverage, sl, tp FROM trades_suggestions ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                logger.info("No trade suggestion found")
                return None
            run_id, side, entry_zone, lev, sl, tp = row
            if side == "NO-TRADE":
                logger.info(f"Last suggestion is NO-TRADE ({run_id})")
                return None
    eng = TradingEngine()
    # Basic sizing: amount is not managed here; assume quote balance allocation out of scope; user should set amount via ENV
    amount = float(os.getenv("EXEC_AMOUNT", "0.001"))
    side_ccxt = "buy" if side == "LONG" else "sell"
    res = eng.open_bracket(symbol, side_ccxt, amount, entry_type=os.getenv("EXEC_ENTRY", "market"), price=None, sl_price=float(sl), tp_price=float(tp), leverage=int(float(lev or 1)))
    logger.info(f"Live executed run_id={run_id}: entry={res.entry_id} sl={res.sl_id} tp={res.tp_id}")
    # Optionally log to DB (live_orders)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO live_orders (exchange, symbol, side, type, amount, price, params, status, exchange_order_id, info) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (os.getenv("EXCHANGE", os.getenv("CCXT_PROVIDER", "binance")), symbol, side_ccxt, os.getenv("EXEC_ENTRY", "market"), amount, None, None, "sent", res.entry_id, None),
                )
    except Exception:
        pass
    return res.entry_id


def main():
    open_from_last_suggestion()


if __name__ == "__main__":
    main()

