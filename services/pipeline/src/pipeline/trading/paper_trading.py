# flake8: noqa
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import ccxt
from loguru import logger

from ..infra.db import get_conn
from ..infra.s3 import upload_bytes
from .publish_telegram import publish_message_to

SYMBOL = os.getenv("PAPER_SYMBOL", "BTC/USDT")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_account(start_equity: float = 1000.0) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Use monthly accounts: cfg_json->>'month' = YYYY-MM
            month = datetime.now(timezone.utc).strftime("%Y-%m")
            cur.execute(
                "SELECT id FROM paper_accounts WHERE cfg_json->>'month'=%s ORDER BY created_at DESC LIMIT 1",
                (month,),
            )
            row = cur.fetchone()
            if row:
                return row[0]
            # Create new monthly account
            cfg = {"month": month}
            cur.execute(
                "INSERT INTO paper_accounts (start_equity, equity, cfg_json) VALUES (%s, %s, %s) RETURNING id",
                (start_equity, start_equity, cfg),
            )
            acc_id = cur.fetchone()[0]
            logger.info(f"Created monthly paper account {acc_id} for {month}")
            return acc_id


def _get_mark_price(symbol: str = SYMBOL) -> float:
    provider = os.getenv("CCXT_PROVIDER", "binance")
    ex = getattr(ccxt, provider)({"enableRateLimit": True})
    # use spot last price for MVP
    t = ex.fetch_ticker(symbol)
    return float(t["last"]) if t and t.get("last") else float(t["close"])  # type: ignore[index]


def _open_from_suggestion(cur, acc_id: str, sug_row: Tuple) -> None:
    run_id = sug_row[0]
    side = sug_row[1]
    if side == "NO-TRADE":
        logger.info(f"Suggestion {run_id} is NO-TRADE")
        return
    cur.execute(
        "SELECT 1 FROM paper_positions WHERE meta_json->>'run_id'=%s AND status='OPEN'",
        (run_id,),
    )
    if cur.fetchone():
        logger.info(f"Position for {run_id} already open")
        return
    price = _get_mark_price()
    entry = float(price)
    sl = float(sug_row[4])
    tp = float(sug_row[5])
    lev = float(sug_row[3])
    # Simple position sizing: 0.5% risk on account equity
    cur.execute("SELECT equity FROM paper_accounts WHERE id=%s", (acc_id,))
    equity = float(cur.fetchone()[0])
    risk_amount = float(os.getenv("PAPER_RISK_PER_TRADE", "0.005")) * equity
    qty = risk_amount / max(1e-6, abs(entry - sl))  # BTC size
    cur.execute(
        "INSERT INTO paper_positions (account_id, opened_at, side, entry, leverage, qty, sl, tp, status, meta_json) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'OPEN', %s) RETURNING pos_id",
        (acc_id, _now_utc(), side, entry, lev, qty, sl, tp, {"run_id": run_id}),
    )
    pos_id = cur.fetchone()[0]
    cur.execute(
        "INSERT INTO paper_trades (pos_id, ts, price, qty, side, fee, reason) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (pos_id, _now_utc(), entry, qty, "OPEN", 0.0, "market_open"),
    )
    logger.info(f"Opened paper position {pos_id} from suggestion {run_id} at {entry}")


def executor_once(run_id: Optional[str] = None):
    """Open paper positions from new trade suggestions.

    If run_id provided — process exactly that suggestion. Otherwise, open all
    suggestions from lookback window with no open positions yet.
    """
    acc_id = _ensure_account()
    with get_conn() as conn:
        with conn.cursor() as cur:
            if run_id:
                cur.execute(
                    "SELECT run_id, side, entry_zone, leverage, sl, tp, times_json FROM trades_suggestions WHERE run_id=%s",
                    (run_id,),
                )
                sug = cur.fetchone()
                if not sug:
                    logger.info("No such suggestion")
                    return
                # Validity check
                try:
                    vt = (
                        (sug[6] or {}).get("valid_until")
                        if isinstance(sug[6], dict)
                        else None
                    )
                    if vt:
                        import dateutil.parser  # type: ignore

                        if _now_utc() > dateutil.parser.isoparse(str(vt)):
                            logger.info(f"Suggestion {run_id} is expired; skipping")
                            return
                except Exception:
                    pass
                _open_from_suggestion(cur, acc_id, sug[:6])
                return
            # Batch open from recent suggestions window
            lookback_days = float(os.getenv("PAPER_EXEC_LOOKBACK_DAYS", "3"))
            cur.execute(
                """
                SELECT run_id, side, entry_zone, leverage, sl, tp
                FROM trades_suggestions
                WHERE created_at >= now() - (%s || ' days')::interval
                  AND (times_json->>'valid_until' IS NULL OR (times_json->>'valid_until')::timestamptz >= now())
                ORDER BY created_at ASC
                """,
                (str(lookback_days),),
            )
            rows = cur.fetchall() or []
            if not rows:
                logger.info("No trade suggestions found in lookback window")
                return
            opened = 0
            for sug in rows:
                _open_from_suggestion(cur, acc_id, sug)
                opened += 1
            logger.info(f"Paper executor processed suggestions: {opened}")


def _close_position(cur, pos_id: str, exit_price: float, reason: str):
    cur.execute(
        "SELECT side, entry, qty FROM paper_positions WHERE pos_id=%s FOR UPDATE",
        (pos_id,),
    )
    side, entry, qty = cur.fetchone()
    side = str(side)
    entry = float(entry)
    qty = float(qty)
    # PnL in USDT
    if side == "LONG":
        pnl = (exit_price - entry) * qty
    else:
        pnl = (entry - exit_price) * qty
    cur.execute("UPDATE paper_positions SET status='CLOSED' WHERE pos_id=%s", (pos_id,))
    cur.execute(
        "INSERT INTO paper_trades (pos_id, ts, price, qty, side, fee, reason) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (pos_id, _now_utc(), exit_price, qty, "CLOSE", 0.0, reason),
    )
    # Update PNL and account equity
    cur.execute(
        "INSERT INTO paper_pnl (pos_id, realized_pnl) VALUES (%s, %s) ON CONFLICT (pos_id) DO UPDATE SET realized_pnl=excluded.realized_pnl",
        (pos_id, pnl),
    )
    cur.execute("SELECT account_id FROM paper_positions WHERE pos_id=%s", (pos_id,))
    account_id = cur.fetchone()[0]
    cur.execute(
        "UPDATE paper_accounts SET equity = equity + %s WHERE id=%s", (pnl, account_id)
    )
    # Record equity curve point
    cur.execute("SELECT equity FROM paper_accounts WHERE id=%s", (account_id,))
    equity = float(cur.fetchone()[0])
    cur.execute(
        "INSERT INTO paper_equity_curve (ts, account_id, equity) VALUES (%s, %s, %s) ON CONFLICT (ts, account_id) DO NOTHING",
        (_now_utc(), account_id, equity),
    )
    # Post‑mortem lesson (best-effort)
    try:
        if os.getenv("POST_MORTEM_ENABLED", "1") in {"1", "true", "True"}:
            cur.execute("SELECT meta_json->>'run_id' FROM paper_positions WHERE pos_id=%s", (pos_id,))
            row = cur.fetchone()
            run_id = row[0] if row and row[0] else None
            context = {
                "run_id": run_id,
                "side": side,
                "entry_price": entry,
                "exit_price": float(exit_price),
                "qty": qty,
                "pnl": pnl,
                "reason": reason,
            }
            from ..agents.post_mortem import run as post_mortem_run, PostMortemInput

            post_mortem_run(PostMortemInput(trade=context))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"post_mortem failed: {e}")


def risk_loop(interval_s: int = 60):
    while True:
        try:
            price = _get_mark_price()
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pos_id, side, sl, tp FROM paper_positions WHERE status='OPEN'"
                    )
                    rows = cur.fetchall()
                    for pos_id, side, sl, tp in rows:
                        side = str(side)
                        sl = float(sl)
                        tp = float(tp)
                        hit_sl = price <= sl if side == "LONG" else price >= sl
                        hit_tp = price >= tp if side == "LONG" else price <= tp
                        if hit_sl:
                            _close_position(cur, pos_id, price, "stop_loss")
                        elif hit_tp:
                            _close_position(cur, pos_id, price, "take_profit")
            # Also sample equity curve every loop
            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT id, equity FROM paper_accounts ORDER BY created_at DESC LIMIT 1"
                        )
                        row = cur.fetchone()
                        if row:
                            acc_id, eq = row
                            cur.execute(
                                "INSERT INTO paper_equity_curve (ts, account_id, equity) VALUES (%s, %s, %s) ON CONFLICT (ts, account_id) DO NOTHING",
                                (_now_utc(), acc_id, float(eq)),
                            )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"equity_curve sample failed: {e}")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"risk_loop error: {e}")
        time.sleep(interval_s)


def executor_loop(interval_s: int = 60):
    while True:
        try:
            executor_once()
        except Exception as e:  # noqa: BLE001
            logger.exception(f"executor_loop error: {e}")
        time.sleep(max(5, int(interval_s)))


def settler_loop(interval_s: int = 60):
    while True:
        try:
            now = _now_utc()
            with get_conn() as conn:
                with conn.cursor() as cur:
                    # join run_id from meta_json to trades_suggestions to get horizon
                    cur.execute(
                        "SELECT p.pos_id, p.side, p.entry, p.qty, p.meta_json->>'run_id' as run_id, t.times_json->>'horizon_ends_at' as horizon_end "
                        "FROM paper_positions p JOIN trades_suggestions t ON (t.run_id = p.meta_json->>'run_id') "
                        "WHERE p.status='OPEN'"
                    )
                    rows = cur.fetchall() or []
            for pos_id, side, entry, qty, run_id, horizon_end in rows:
                if not horizon_end:
                    continue
                try:
                    import dateutil.parser  # type: ignore[import-not-found, import-untyped]

                    horizon_dt = dateutil.parser.isoparse(horizon_end)
                except Exception:
                    continue
                if now >= horizon_dt:
                    price = _get_mark_price()
                    _close_position(cur, pos_id, price, "horizon_close")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"settler_loop error: {e}")
        time.sleep(interval_s)


def admin_report():
    admin_chat = os.getenv("TELEGRAM_ADMIN_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if not admin_chat:
        logger.warning("No admin chat configured; skipping report")
        return
    from datetime import timedelta

    now = _now_utc()
    day_ago = now - timedelta(days=1)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, equity, start_equity FROM paper_accounts ORDER BY created_at DESC LIMIT 1"
            )
            acc = cur.fetchone()
            if not acc:
                publish_message_to(admin_chat, "paper_account_not_found")
                return
            acc_id, equity, start_eq = acc
            cur.execute(
                "SELECT COALESCE(SUM(realized_pnl),0) FROM paper_pnl WHERE pos_id IN (SELECT pos_id FROM paper_positions WHERE opened_at >= %s)",
                (day_ago,),
            )
            pnl_day = float(cur.fetchone()[0] or 0.0)
            cur.execute(
                "SELECT COUNT(*) FROM paper_positions WHERE opened_at >= %s AND status='CLOSED'",
                (day_ago,),
            )
            closed = int(cur.fetchone()[0])
            text = (
                f"<b>Paper: дневной отчёт</b>\n"
                f"Equity: {equity:.2f} (start {start_eq:.2f})\n"
                f"PnL 24h: {pnl_day:.2f}\n"
                f"Сделок закрыто: {closed}"
            )
            publish_message_to(admin_chat, text)


def admin_weekly_report():
    admin_chat = os.getenv("TELEGRAM_ADMIN_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if not admin_chat:
        logger.warning("No admin chat configured; skipping weekly report")
        return
    import io
    from datetime import timedelta

    import matplotlib.pyplot as plt

    now = _now_utc()
    week_ago = now - timedelta(days=7)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, start_equity FROM paper_accounts ORDER BY created_at ASC LIMIT 1"
            )
            acc_row = cur.fetchone()
            if not acc_row:
                publish_message_to(admin_chat, "paper_account_not_found")
                return
            acc_id, start_eq = acc_row
            cur.execute(
                "SELECT ts, equity FROM paper_equity_curve WHERE account_id=%s AND ts >= %s ORDER BY ts ASC",
                (acc_id, week_ago),
            )
            rows = cur.fetchall() or []
            if not rows:
                publish_message_to(admin_chat, "paper_no_equity")
                return
            ts = [r[0] for r in rows]
            eq = [float(r[1]) for r in rows]
    # Plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(ts, eq, label="Equity")
    ax.set_title("Paper Equity (7d)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    # Upload chart
    path = f"paper_reports/{now.date().isoformat()}/equity_7d.png"
    s3_uri = upload_bytes(path, buf.getvalue(), content_type="image/png")
    from .publish_telegram import publish_photo_from_s3

    publish_photo_from_s3(
        s3_uri, caption="paper_equity_week", days=7, start_eq=start_eq
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("executor")
    el = sub.add_parser("executor_loop")
    el.add_argument("--interval", type=int, default=60)
    rl = sub.add_parser("risk")
    rl.add_argument("--interval", type=int, default=60)
    sl = sub.add_parser("settler")
    sl.add_argument("--interval", type=int, default=60)
    sub.add_parser("admin_report")
    args = parser.parse_args()

    if args.cmd == "executor":
        executor_once()
    elif args.cmd == "executor_loop":
        executor_loop(interval_s=args.interval)
    elif args.cmd == "risk":
        risk_loop(interval_s=args.interval)
    elif args.cmd == "settler":
        settler_loop(interval_s=args.interval)
    elif args.cmd == "admin_report":
        admin_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
