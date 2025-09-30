from __future__ import annotations

"""Manual utility to adjust agent weights per regime.

Usage:
    PYTHONPATH=services/pipeline/src python services/pipeline/ops/update_agent_performance.py \
        --agent smc --regime trend_up --wins 12 --losses 8 --weight 1.15

You can also provide a CSV file with columns agent_name,regime_label,wins,losses,weight.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

from pipeline.infra.db import upsert_agent_performance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update agent performance weights")
    parser.add_argument("--agent", help="Agent name (e.g. smc, advanced_ta)")
    parser.add_argument("--regime", help="Regime label (trend_up/trend_down/range)")
    parser.add_argument("--wins", type=int, default=0)
    parser.add_argument("--losses", type=int, default=0)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--csv", type=Path, help="Optional CSV file with multiple rows")
    return parser.parse_args()


def update_from_row(row: dict[str, str]) -> None:
    agent = row.get("agent") or row.get("agent_name")
    regime = row.get("regime") or row.get("regime_label")
    wins = int(row.get("wins", "0"))
    losses = int(row.get("losses", "0"))
    weight = float(row.get("weight", "1.0"))
    if not agent or not regime:
        raise ValueError(f"Invalid row: {row}")
    upsert_agent_performance(agent, regime, wins, losses, weight)
    print(f"Updated {agent} / {regime}: wins={wins}, losses={losses}, weight={weight}")


def main() -> None:
    args = parse_args()
    if args.csv:
        with args.csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                update_from_row(row)
        return
    if not args.agent or not args.regime:
        raise SystemExit("Specify --agent and --regime or use --csv")
    row = {
        "agent": args.agent,
        "regime": args.regime,
        "wins": str(args.wins),
        "losses": str(args.losses),
        "weight": str(args.weight),
    }
    update_from_row(row)


if __name__ == "__main__":
    if "DATABASE_URL" not in os.environ:
        raise SystemExit("DATABASE_URL is required")
    main()
