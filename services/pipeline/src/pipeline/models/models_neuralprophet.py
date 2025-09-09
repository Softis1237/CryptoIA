from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from loguru import logger


def try_add_np_preds(close_5m: pd.Series, horizon_steps: int, last_price: float, atr: float, horizon_minutes: int) -> List[dict]:
    preds: List[dict] = []
    if os.getenv("USE_NEURALPROPHET", "0") not in {"1", "true", "True"}:
        return preds
    try:
        from neuralprophet import NeuralProphet  # type: ignore
    except Exception as e:  # noqa: BLE001
        logger.warning(f"NeuralProphet not available: {e}")
        return preds

    try:
        df = pd.DataFrame({"ds": close_5m.index.to_pydatetime(), "y": close_5m.values})
        epochs = int(os.getenv("NP_EPOCHS", "5"))
        m = NeuralProphet(n_forecasts=1, n_lags=min(72, max(12, horizon_steps * 2)))
        m.fit(df, freq="5min", epochs=epochs)
        future = m.make_future_dataframe(df, periods=horizon_steps)
        fc = m.predict(future)
        y_hat = float(fc["yhat1"].iloc[-1])
        # interval from ATR heuristic
        scale = max(1.0, np.sqrt(horizon_minutes / 60.0))
        band = float(atr * 1.5 * scale)
        pi_low = y_hat - band
        pi_high = y_hat + band
        proba_up = 0.5 + np.tanh((y_hat - last_price) / (atr + 1e-8)) * 0.25
        preds.append({
            "model": "neuralprophet",
            "y_hat": y_hat,
            "pi_low": pi_low,
            "pi_high": pi_high,
            "proba_up": float(proba_up),
            "cv_metrics": {},
        })
    except Exception as e:  # noqa: BLE001
        logger.warning(f"NeuralProphet failed: {e}")
    return preds
