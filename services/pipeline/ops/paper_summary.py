from __future__ import annotations

"""Generate summary stats for paper trading PnL."""

import argparse
import os
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2.extras import RealDictCursor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper trading summary")
    parser.add_argument("--days", type=int, default=30)
    return parser.parse_args()


def _connect():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set")
    return psycopg2.connect(dsn)


def main() -> None:
    args = parse_args()
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    sql = (
        "SELECT pnl, created_at FROM paper_trades WHERE created_at >= %s"
    )
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (since,))
            rows = cur.fetchall()
    if not rows:
        print("No paper trades for the window")
        return
    pnls = [float(r.get("pnl") or 0.0) for r in rows]
    total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    print(f"Trades: {len(pnls)}")
    print(f"Total PnL: {total:.2f}")
    if pnls:
        print(f"Average PnL: {total/len(pnls):.2f}")
    print(f"Wins: {wins} Losses: {losses}")


if __name__ == "__main__":
    main()
