from __future__ import annotations

from typing import Tuple


def calibrate_proba_by_uncertainty(proba_up: float, interval: Tuple[float, float], price: float, atr: float | None = None) -> float:
    """Simple uncertainty-aware calibration.

    - Compresses probability toward 0.5 when interval is wide relative to price (uncertain).
    - Slightly expands when interval is narrow.
    - ATR can be used to normalize width; if provided and width >> ATR, compress more.
    """
    low, high = float(interval[0]), float(interval[1])
    width = max(0.0, high - low)
    rel_w = width / max(1e-6, price)
    factor = 1.0
    # baseline compression: 1% width -> ~0.9; 0.2% width -> ~1.05
    if rel_w >= 0.02:
        factor = 0.7
    elif rel_w >= 0.01:
        factor = 0.85
    elif rel_w <= 0.003:
        factor = 1.05
    # ATR-based adjustment
    if atr is not None and atr > 0:
        atr_ratio = width / atr
        if atr_ratio > 8:
            factor *= 0.85
        elif atr_ratio < 2:
            factor *= 1.03

    # apply compression around 0.5
    delta = proba_up - 0.5
    calibrated = 0.5 + delta * factor
    # clamp
    return float(max(0.01, min(0.99, calibrated)))
