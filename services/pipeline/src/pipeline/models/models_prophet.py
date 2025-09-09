from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from loguru import logger


def try_add_prophet_preds(close_5m: pd.Series, horizon_steps: int, last_price: float, atr: float, horizon_minutes: int) -> List[dict]:
    preds: List[dict] = []
    use_fb = os.getenv("USE_PROPHET", "0") in {"1", "true", "True"}
    use_darts = os.getenv("USE_DARTS_PROPHET", "0") in {"1", "true", "True"}
    if not (use_fb or use_darts):
        return preds
    # Prophet requires a dataframe with columns ds, y
    try:
        if use_fb:
            try:
                from prophet import Prophet  # type: ignore
            except Exception:
                from fbprophet import Prophet  # type: ignore  # legacy
            df = pd.DataFrame({"ds": close_5m.index.to_pydatetime(), "y": close_5m.values})
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
            m.fit(df)
            future = m.make_future_dataframe(periods=horizon_steps, freq="5min")
            fc = m.predict(future)
            y_hat = float(fc["yhat"].iloc[-1])
        elif use_darts:
            from darts import TimeSeries
            from darts.models import Prophet as DartsProphet

            ts = TimeSeries.from_series(close_5m)
            model = DartsProphet()
            model.fit(ts)
            fc = model.predict(horizon_steps)
            y_hat = float(fc.values()[-1, 0])
        else:
            return preds

        # Interval heuristic from ATR
        scale = max(1.0, np.sqrt(horizon_minutes / 60.0))
        band = float(atr * 1.5 * scale)
        pi_low = y_hat - band
        pi_high = y_hat + band
        proba_up = 0.5 + np.tanh((y_hat - last_price) / (atr + 1e-8)) * 0.25
        preds.append({
            "model": "prophet",
            "y_hat": y_hat,
            "pi_low": pi_low,
            "pi_high": pi_high,
            "proba_up": float(proba_up),
            "cv_metrics": {},
        })
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Prophet not available or failed: {e}")
    return preds
