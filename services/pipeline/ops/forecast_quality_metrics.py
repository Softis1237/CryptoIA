from __future__ import annotations

"""Push forecast quality metrics (MAE/SMAPE/DA) to Prometheus."""

import argparse
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from pipeline.infra.metrics import push_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push forecast quality metrics")
    parser.add_argument("--hours", type=int, default=168, help="Lookback window in hours")
    parser.add_argument("--horizon", default="4h", help="Prediction horizon")
    return parser.parse_args()


def _connect():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set")
    return psycopg2.connect(dsn)


def compute_metrics(hours: int, horizon: str) -> dict[str, float]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    sql = (
        "SELECT p.y_hat, p.per_model_json, po.y_true, po.created_at "
        "FROM predictions p JOIN prediction_outcomes po USING (run_id, horizon) "
        "WHERE p.horizon=%s AND po.created_at >= %s"
    )
    maes, smapes, das = [], [], []
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (horizon, since))
            rows = cur.fetchall()
            for row in rows:
                y_hat = float(row.get("y_hat") or 0.0)
                y_true = row.get("y_true")
                if y_true is None:
                    continue
                y_true = float(y_true)
                maes.append(abs(y_hat - y_true))
                denom = abs(y_hat) + abs(y_true) + 1e-9
                smapes.append(abs(y_hat - y_true) / denom)
                # directional accuracy: based on per_model_json if available
                try:
                    per_model = row.get("per_model_json") or {}
                    prev = float(per_model.get("baseline", {}).get("prev_close", y_true))
                    actual_dir = np.sign(y_true - prev)
                    pred_dir = np.sign(y_hat - prev)
                    das.append(1.0 if actual_dir == pred_dir else 0.0)
                except Exception:
                    continue
    if not maes:
        return {}
    return {
        "mae": float(np.mean(maes)),
        "smape": float(np.mean(smapes)),
        "da": float(np.mean(das)) if das else 0.0,
        "count": float(len(maes)),
    }


def main() -> None:
    args = parse_args()
    metrics = compute_metrics(args.hours, args.horizon)
    if not metrics:
        print("No data for metrics")
        return
    push_values(
        job="forecast_quality",
        values={
            "mae": metrics["mae"],
            "smape": metrics["smape"],
            "da": metrics["da"],
            "count": metrics["count"],
        },
        labels={"horizon": args.horizon, "window_hours": str(args.hours)},
    )
    print(f"Pushed forecast metrics for horizon={args.horizon} window={args.hours}h")


if __name__ == "__main__":
    main()
