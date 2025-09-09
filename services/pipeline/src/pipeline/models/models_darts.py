from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

def _interval_from_vol(last_price: float, atr: float, horizon_minutes: int) -> tuple[float, float]:
    import numpy as np

    scale = max(1.0, np.sqrt(horizon_minutes / 60.0))
    band = float(atr * 1.5 * scale)
    return last_price - band, last_price + band


def try_add_darts_preds(close_5m: pd.Series, horizon_steps: int, last_price: float, atr: float, horizon_minutes: int) -> List[dict]:
    preds: List[dict] = []
    if os.getenv("USE_DARTS", "0") not in {"1", "true", "True"}:
        return preds
    try:
        from darts import TimeSeries
        from darts.models import Theta  # lightweight; avoids heavy torch

        ts = TimeSeries.from_series(close_5m)
        model = Theta()
        model.fit(ts)
        fc = model.predict(horizon_steps)
        y_hat = float(fc.values()[-1, 0])
        pi_low, pi_high = _interval_from_vol(y_hat, atr, horizon_minutes)
        # naive CV: 1-step rolling
        # We avoid heavy backtest; use simple sMAPE vs naive last value
        true_tail = close_5m.iloc[-60:].values
        pred_tail = np.repeat(close_5m.iloc[-61], min(60, len(true_tail)))
        smape = float(100.0 * np.mean(np.abs(pred_tail - true_tail) / (np.abs(pred_tail) + np.abs(true_tail) + 1e-9)))
        proba_up = 0.5 + np.tanh((y_hat - last_price) / (atr + 1e-8)) * 0.25
        preds.append({
            "model": "theta",
            "y_hat": y_hat,
            "pi_low": pi_low,
            "pi_high": pi_high,
            "proba_up": float(proba_up),
            "cv_metrics": {"smape": smape},
        })
        # Optional N-BEATS (heavy; requires torch). Enable with USE_DARTS_NBEATS=1
        if os.getenv("USE_DARTS_NBEATS", "0") in {"1", "true", "True"}:
            try:
                from darts.models import NBEATSModel

                nbeats = NBEATSModel(
                    input_chunk_length=min(240, max(24, horizon_steps * 3)),
                    output_chunk_length=horizon_steps,
                    n_epochs=int(os.getenv("NBEATS_EPOCHS", "5")),
                    random_state=42,
                    torch_device="cpu",
                )
                nbeats.fit(ts)
                fc2 = nbeats.predict(horizon_steps)
                y2 = float(fc2.values()[-1, 0])
                lo2, hi2 = _interval_from_vol(y2, atr, horizon_minutes)
                proba2 = 0.5 + np.tanh((y2 - last_price) / (atr + 1e-8)) * 0.25
                preds.append({
                    "model": "nbeats",
                    "y_hat": y2,
                    "pi_low": lo2,
                    "pi_high": hi2,
                    "proba_up": float(proba2),
                    "cv_metrics": {"smape": smape},
                })
            except Exception as e:
                from loguru import logger
                logger.warning(f"Darts N-BEATS not available or failed: {e}")
        # Optional TFT (Temporal Fusion Transformer)
        if os.getenv("USE_DARTS_TFT", "0") in {"1", "true", "True"}:
            try:
                from darts.models import TFTModel
                from darts import TimeSeries
                ts = TimeSeries.from_series(close_5m)
                tft = TFTModel(
                    input_chunk_length=min(288, max(48, horizon_steps * 6)),
                    output_chunk_length=horizon_steps,
                    random_state=42,
                    n_epochs=int(os.getenv("TFT_EPOCHS", "5")),
                    torch_device="cpu",
                )
                tft.fit(ts)
                fc3 = tft.predict(horizon_steps)
                y3 = float(fc3.values()[-1, 0])
                lo3, hi3 = _interval_from_vol(y3, atr, horizon_minutes)
                proba3 = 0.5 + np.tanh((y3 - last_price) / (atr + 1e-8)) * 0.25
                preds.append({
                    "model": "tft",
                    "y_hat": y3,
                    "pi_low": lo3,
                    "pi_high": hi3,
                    "proba_up": float(proba3),
                    "cv_metrics": {"smape": smape},
                })
            except Exception as e:
                from loguru import logger
                logger.warning(f"Darts TFT not available or failed: {e}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Darts not available or failed: {e}")
    return preds
