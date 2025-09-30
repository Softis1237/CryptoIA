from __future__ import annotations

"""Recompute agent performance weights based on trade outcomes.

Usage:
    PYTHONPATH=services/pipeline/src python services/pipeline/ops/recompute_agent_performance.py \
        --days 180 --regime trend_up

The script aggregates wins/losses for each agent per regime using `trades_outcomes`
(must include columns: run_id, regime_label, agents_involved, pnl) and writes results
into `agent_performance` (migration 045).
"""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable

import psycopg2
from psycopg2.extras import RealDictCursor

from pipeline.infra.db import upsert_agent_performance


@dataclass(slots=True)
class AgentStats:
    wins: int = 0
    losses: int = 0

    def update(self, pnl: float) -> None:
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

    def to_weight(self) -> float:
        total = self.wins + self.losses
        if total == 0:
            return 1.0
        winrate = self.wins / total
        # Map [0..1] -> [0.7..1.3]
        return max(0.7, min(1.3, 0.7 + winrate * 0.6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute agent performance weights")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--regime", help="Optional regime filter", default=None)
    return parser.parse_args()


def _connect():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set")
    return psycopg2.connect(dsn)


def _fetch_trades(cur: RealDictCursor, since: datetime, regime: str | None):
    sql = (
        "SELECT run_id, regime_label, agents_involved, pnl "
        "FROM trade_outcomes WHERE executed_at >= %s"
    )
    params = [since]
    if regime:
        sql += " AND regime_label=%s"
        params.append(regime)
    cur.execute(sql, params)
    return cur.fetchall()


def recompute(days: int, regime: str | None) -> None:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    stats: Dict[tuple[str, str], AgentStats] = {}
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for row in _fetch_trades(cur, since, regime):
                regime_label = row.get("regime_label") or "unknown"
                pnl = float(row.get("pnl") or 0.0)
                agents = row.get("agents_involved") or []
                if isinstance(agents, str):
                    agents = [agents]
                for agent in agents:
                    key = (agent, regime_label)
                    stats.setdefault(key, AgentStats()).update(pnl)
    if not stats:
        print("No trades found; nothing to update")
        return
    for (agent, regime_label), aggregation in stats.items():
        weight = aggregation.to_weight()
        upsert_agent_performance(agent, regime_label, aggregation.wins, aggregation.losses, weight)
        print(f"Updated {agent}/{regime_label}: wins={aggregation.wins} losses={aggregation.losses} weight={weight:.3f}")


def main() -> None:
    args = parse_args()
    recompute(args.days, args.regime)


if __name__ == "__main__":
    if "DATABASE_URL" not in os.environ:
        raise SystemExit("DATABASE_URL is required")
    main()
