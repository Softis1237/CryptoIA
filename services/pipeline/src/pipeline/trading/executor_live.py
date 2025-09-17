from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from ..infra.db import get_conn
from .live_portfolio_manager import LivePortfolioManager
from .engine import TradingEngine


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _compute_dynamic_size(entry: float, sl: float, risk_pct: float, equity: float) -> float:
    dist = abs(float(entry) - float(sl))
    if dist <= 1e-9:
        return 0.0
    risk_amt = max(0.0, float(risk_pct)) * max(0.0, float(equity))
    return round(risk_amt / dist, 6)


def open_from_last_suggestion(
    symbol: str = os.getenv("EXEC_SYMBOL", "BTC/USDT")
) -> Optional[str]:
    if os.getenv("EXECUTE_LIVE", "0") not in {"1", "true", "True"}:
        logger.info("EXECUTE_LIVE=0: skip live execution")
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, side, entry_zone, leverage, sl, tp FROM trades_suggestions ORDER BY created_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            if not row:
                logger.info("No trade suggestion found")
                return None
            run_id, side, entry_zone, lev, sl, tp = row
            if side == "NO-TRADE":
                logger.info(f"Last suggestion is NO-TRADE ({run_id})")
                return None
    eng = TradingEngine()
    # Dynamic sizing (optional)
    use_dyn = os.getenv("EXEC_DYNAMIC_SIZE", "1") in {"1", "true", "True"}
    risk_pct = float(os.getenv("EXEC_RISK_PER_TRADE", "0.01"))
    amount = float(os.getenv("EXEC_AMOUNT", "0.001"))
    if use_dyn:
        try:
            lpm = LivePortfolioManager()
            eq = lpm.total_equity()
            # use mid of entry zone if present
            try:
                entry_mid = float(entry_zone[0] if isinstance(entry_zone, (list, tuple)) else None)  # type: ignore[name-defined]
            except Exception:
                entry_mid = None
            if entry_mid is None:
                entry_mid = float(os.getenv("EXEC_ENTRY_PRICE_OVERRIDE", "0")) or None
            if entry_mid is not None and sl is not None:  # type: ignore[name-defined]
                dyn = _compute_dynamic_size(entry_mid, float(sl), risk_pct, eq)  # type: ignore[arg-type]
                if dyn > 0:
                    amount = dyn
        except Exception:
            pass
    side_ccxt = "buy" if side == "LONG" else "sell"
    # Add retries for exchange flakiness
    def _do_open():
        return eng.open_bracket(
            symbol,
            side_ccxt,
            amount,
            entry_type=os.getenv("EXEC_ENTRY", "market"),
            price=None,
            sl_price=float(sl),
            tp_price=float(tp),
            leverage=int(float(lev or 1)),
        )
    try:
        from tenacity import retry, wait_exponential, stop_after_attempt  # type: ignore

        @retry(wait=wait_exponential(multiplier=1, max=10), stop=stop_after_attempt(3))
        def _open_retry():
            return _do_open()

        res = _open_retry()
    except Exception:
        res = _do_open()
    logger.info(
        f"Live executed run_id={run_id}: entry={res.entry_id} sl={res.sl_id} tp={res.tp_id}"
    )
    # Optionally log to DB (live_orders)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO live_orders (exchange, symbol, side, type, amount, price, params, status, exchange_order_id, info) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        os.getenv("EXCHANGE", os.getenv("CCXT_PROVIDER", "binance")),
                        symbol,
                        side_ccxt,
                        os.getenv("EXEC_ENTRY", "market"),
                        amount,
                        None,
                        None,
                        "sent",
                        res.entry_id,
                        None,
                    ),
                )
    except Exception as e:
        logger.exception("Failed to record live order in database")
        # admin alert (best-effort)
        try:
            from ..telegram_bot.publisher import publish_message

            admin = os.getenv("TELEGRAM_ADMIN_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
            if admin:
                publish_message(f"Live order DB record failed: {e}")
        except Exception:
            pass
    return res.entry_id


def main():
    open_from_last_suggestion()


if __name__ == "__main__":
    main()
