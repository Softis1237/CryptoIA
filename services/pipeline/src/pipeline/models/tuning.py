from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def _load_close_from_features(s3_uri: str) -> pd.Series:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ..infra.s3 import download_bytes

    raw = download_bytes(s3_uri)
    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas().sort_values("ts")
    dt = pd.to_datetime(df["ts"], unit="s", utc=True)
    return pd.Series(df["close"].astype(float).values, index=dt)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return float(100.0 * np.mean(np.abs(y_pred - y_true) / denom))


def tune_arima(series: pd.Series, horizon_steps: int = 24, window: int = 100) -> Tuple[Tuple[int, int, int], float]:
    """Small Optuna search for ARIMA(p,d,q) minimizing sMAPE on one‑step rolling."""
    try:
        import optuna  # type: ignore
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Optuna/ARIMA unavailable: {e}")
        return (1, 1, 1), 999.0

    def objective(trial: "optuna.Trial") -> float:
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 3)
        n = len(series)
        y_true = []
        y_pred = []
        for idx in range(max(20, n - window), n):
            train = series.iloc[:idx]
            try:
                fit = ARIMA(train, order=(p, d, q)).fit()
                pred = float(fit.forecast(1).iloc[-1])
            except Exception:
                pred = float(train.iloc[-1])
            y_true.append(float(series.iloc[idx]))
            y_pred.append(pred)
        return _smape(np.asarray(y_true), np.asarray(y_pred))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(os.getenv("ARIMA_TUNING_TRIALS", "20")), show_progress_bar=False)
    p = int(study.best_params.get("p", 1))
    d = int(study.best_params.get("d", 1))
    q = int(study.best_params.get("q", 1))
    return (p, d, q), float(study.best_value)


def log_mlflow(run_name: str, params: Dict[str, float], metrics: Dict[str, float]) -> None:
    tracking = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking:
        return
    try:
        import mlflow  # type: ignore

        mlflow.set_tracking_uri(tracking)
        with mlflow.start_run(run_name=run_name):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"MLflow logging skipped: {e}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="S3 URI of features.parquet")
    ap.add_argument("--horizon_minutes", type=int, default=240)
    args = ap.parse_args()

    series = _load_close_from_features(args.features).resample("5min").last().dropna()
    steps = max(1, args.horizon_minutes // 5)
    (p, d, q), smape = tune_arima(series, horizon_steps=steps)
    log_mlflow(run_name="arima_tuning", params={"p": p, "d": d, "q": q}, metrics={"smape": smape})
    print(f"best_order=({p},{d},{q}) smape={smape:.4f}")


if __name__ == "__main__":
    main()
