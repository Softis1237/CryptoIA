from __future__ import annotations

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import dump
from sklearn.linear_model import LogisticRegression

from ..infra.s3 import download_bytes


def _build_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Compute features for each row similar to predictor
    close = df["close"].astype(float)
    ema20 = df.get("ema_20", close.ewm(span=20, adjust=False).mean()).astype(float)
    ema50 = df.get("ema_50", close.ewm(span=50, adjust=False).mean()).astype(float)
    atr = (
        df.get("atr_14").astype(float)
        if "atr_14" in df.columns
        else (df["high"] - df["low"]).rolling(14).mean().fillna(method="bfill").astype(float)
    )
    trend = ((ema20 - ema50) / (close.abs() + 1e-9)).ewm(span=20, adjust=False).mean()
    vol = (atr / (close.abs() + 1e-9)) * 100.0
    ret_30 = close.pct_change(30).fillna(0.0) * 100.0

    X = np.column_stack([trend.values, vol.values, ret_30.values])

    # Generate labels using heuristic logits
    logit_up = trend * 800 - vol * 0.5
    logit_down = -trend * 800 - vol * 0.5
    logit_range = -(trend.abs() * 600) - vol * 0.2
    logit_vol = vol * 0.8 - trend.abs() * 200
    vol_penalty = -((np.maximum(0.0, vol - 2.0)) ** 2) * 0.1
    logit_healthy = trend * 900 + vol_penalty
    logit_crash = np.maximum(0.0, -ret_30) * 2.5 + np.maximum(0.0, vol - 2.0) * 0.5 - np.maximum(0.0, trend) * 300
    logits = np.column_stack(
        [logit_up, logit_down, logit_range, logit_vol, logit_healthy, logit_crash]
    )
    y = np.argmax(logits.values, axis=1)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regime classifier")
    parser.add_argument("features_s3", help="S3 path to training features")
    parser.add_argument("out_path", help="Where to save the trained model")
    args = parser.parse_args()

    raw = download_bytes(args.features_s3)
    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas().sort_values("ts").reset_index(drop=True)

    X, y = _build_dataset(df)
    model = LogisticRegression(max_iter=1000, multi_class="ovr")
    model.fit(X, y)
    dump(model, args.out_path)


if __name__ == "__main__":
    main()
