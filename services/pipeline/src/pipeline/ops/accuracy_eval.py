from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..infra.db import get_conn


@dataclass
class AccuracyReport:
    rmse: float
    mae: float
    direction_accuracy: float
    coverage: float
    brier: float | None
    calibration: List[Tuple[float, float]]


def _load_pairs(horizon: str) -> pd.DataFrame:
    sql = (
        "SELECT p.y_hat, p.pi_low, p.pi_high, p.proba_up, o.y_true, o.direction_correct "
        "FROM predictions p JOIN prediction_outcomes o ON p.run_id=o.run_id AND p.horizon=o.horizon "
        "WHERE p.horizon=%s AND o.y_true IS NOT NULL"
    )
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=(horizon,))


def compute_accuracy(horizon: str) -> AccuracyReport | None:
    df = _load_pairs(horizon)
    if df.empty:
        logger.warning(f"no data for horizon {horizon}")
        return None
    err = df["y_hat"] - df["y_true"]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    direction_accuracy = float(np.mean(df["direction_correct"].astype(float)))
    coverage = float(
        np.mean((df["y_true"] >= df["pi_low"]) & (df["y_true"] <= df["pi_high"]))
    )
    brier = None
    calibration: List[Tuple[float, float]] = []
    if "proba_up" in df.columns and not df["proba_up"].isna().all():
        proba = df["proba_up"].astype(float)
        pred_up = proba >= 0.5
        actual_up = np.where(df["direction_correct"], pred_up, ~pred_up).astype(float)
        brier = float(np.mean((proba - actual_up) ** 2))
        bins = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(proba, bins) - 1
        for b in range(10):
            m = bin_ids == b
            if np.any(m):
                calibration.append((float(proba[m].mean()), float(actual_up[m].mean())))
    return AccuracyReport(rmse, mae, direction_accuracy, coverage, brier, calibration)


def main() -> Dict[str, AccuracyReport]:
    results: Dict[str, AccuracyReport] = {}
    for hz in ("4h", "12h"):
        r = compute_accuracy(hz)
        if r:
            results[hz] = r
            logger.info(f"accuracy {hz}: {r}")
    return results


if __name__ == "__main__":  # pragma: no cover
    main()
