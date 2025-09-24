from __future__ import annotations

"""Simple stacking meta-learner using recent matured predictions.

Learns linear weights to combine per-model predictions into y_true, using
recent rows from `predictions` table (via fetch_predictions_for_cv) and
exchange prices as ground truth (same logic as ModelsCVAgent).

No external ML deps: uses numpy lstsq with small L2 regularization and
non-negativity clipping + renormalization.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from loguru import logger

from ..infra.db import fetch_predictions_for_cv
from ..utils.horizons import horizon_to_timedelta, horizon_to_hours


def _get_y_true_ccxt(ts_iso: str, horizon: str, provider: str = "binance") -> Optional[float]:
    try:
        import ccxt  # type: ignore
        ex = getattr(ccxt, provider)({"enableRateLimit": True})
        ts = pd.Timestamp(ts_iso, tz="UTC").to_pydatetime()
        ahead = ts + horizon_to_timedelta(horizon)
        market = "BTC/USDT"
        ohlcv = ex.fetch_ohlcv(market, timeframe="1m", since=int((ahead.timestamp() - 60 * 5) * 1000), limit=10)
        if not ohlcv:
            return None
        tgt = int(ahead.timestamp() * 1000)
        row = min(ohlcv, key=lambda r: abs(r[0] - tgt))
        return float(row[4])
    except Exception:
        return None


def _solve_nnls(X: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> np.ndarray:
    # Add L2 ridge term: solve (X^T X + l2 I) w = X^T y
    XtX = X.T @ X
    XtX += l2 * np.eye(X.shape[1])
    Xty = X.T @ y
    try:
        w = np.linalg.solve(XtX, Xty)
    except Exception:
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
    # Non-negative clipping and renorm
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        # fallback to uniform
        w = np.ones_like(w) / max(1, len(w))
    else:
        w = w / s
    return w


def suggest_weights(horizon: str, provider: str = "binance", min_rows: int = 30) -> Tuple[Dict[str, float], int]:
    """Return stacking weights per model name and number of samples used.

    Uses matured predictions from DB and exchange prices as ground truth.
    """
    min_age_hours = max(1.0, horizon_to_hours(horizon))
    rows = fetch_predictions_for_cv(horizon, min_age_hours=min_age_hours)
    if not rows:
        return {}, 0
    # Collect per-model y_hat and y_true
    data = []
    model_names: List[str] = []
    for run_id, hz, created_at, y_hat, pi_low, pi_high, per_model in rows:
        yt = _get_y_true_ccxt(created_at, horizon=horizon, provider=provider)
        if yt is None:
            continue
        # Build row of per-model preds
        if not model_names:
            model_names = sorted(list(per_model.keys()))
        x = []
        for m in model_names:
            try:
                x.append(float((per_model.get(m) or {}).get("y_hat", 0.0)))
            except Exception:
                x.append(0.0)
        data.append((x, float(yt)))
    if len(data) < min_rows:
        return {}, len(data)
    X = np.asarray([r[0] for r in data], dtype=float)
    y = np.asarray([r[1] for r in data], dtype=float)
    w = _solve_nnls(X, y, l2=1e-3)
    weights = {m: float(w[i]) for i, m in enumerate(model_names)}
    return weights, len(data)


def combine_with_weights(preds: List[dict], weights: Dict[str, float]) -> Tuple[float, float, float, float]:
    """Combine provided model preds with precomputed weights.

    Returns (y_hat, pi_low, pi_high, proba_up).
    Unknown models get zero weight; if total=0, fall back to uniform.
    """
    total = 0.0
    wmap: Dict[str, float] = {}
    for p in preds:
        m = str(p.get("model"))
        w = float(weights.get(m, 0.0))
        wmap[m] = w
        total += w
    if total <= 0:
        # uniform fallback
        wmap = {str(p.get("model")): 1.0 for p in preds}
        total = float(len(preds))
    for k in list(wmap.keys()):
        wmap[k] = wmap[k] / total
    def wavg(key: str) -> float:
        return sum(float(p.get(key, 0.0)) * wmap.get(str(p.get("model")), 0.0) for p in preds)
    return wavg("y_hat"), wavg("pi_low"), wavg("pi_high"), wavg("proba_up")
