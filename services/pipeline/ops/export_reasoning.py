from __future__ import annotations

"""Export reasoning/critique logs for prompt tuning."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import psycopg2
from psycopg2.extras import RealDictCursor


def _connect():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set")
    return psycopg2.connect(dsn)


def _fetch(cur: RealDictCursor, limit: int, mode: str | None = None):
    if mode:
        cur.execute(
            "SELECT * FROM arbiter_reasoning WHERE mode=%s ORDER BY created_at DESC LIMIT %s",
            (mode, limit),
        )
    else:
        cur.execute(
            "SELECT * FROM arbiter_reasoning ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
    return cur.fetchall()


def export(limit: int, mode: str | None, include_critique: bool, output: Path) -> None:
    rows: list[Dict[str, Any]] = []
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            rows = _fetch(cur, limit=limit, mode=mode)
            if include_critique:
                ids = [row["run_id"] for row in rows]
                if ids:
                    cur.execute(
                        "SELECT * FROM arbiter_selfcritique WHERE run_id = ANY(%s)", (ids,)
                    )
                    critique_map = {row["run_id"]: row for row in cur.fetchall()}
                    for row in rows:
                        row["critique"] = critique_map.get(row["run_id"])
    payload = {"items": rows, "count": len(rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Exported {len(rows)} entries -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CoT reasoning for analysis")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--mode", choices=["legacy", "modern"], default=None)
    parser.add_argument("--include-critique", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reasoning_export.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export(args.limit, args.mode, args.include_critique, args.output)


if __name__ == "__main__":
    main()
