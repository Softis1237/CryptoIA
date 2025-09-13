from __future__ import annotations

from typing import Tuple, List

import os
from loguru import logger

try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except Exception:  # pragma: no cover
    IsotonicRegression = None  # type: ignore

try:
    import redis as _redis  # type: ignore
except Exception:  # pragma: no cover
    _redis = None  # type: ignore

try:
    from ..infra.db import get_conn  # type: ignore
except Exception:  # pragma: no cover
    get_conn = None  # type: ignore


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


def _redis_client():
    if _redis is None:
        return None
    try:
        host = os.getenv("REDIS_HOST", "redis")
        port = int(os.getenv("REDIS_PORT", "6379"))
        return _redis.Redis(host=host, port=port, decode_responses=True)
    except Exception:
        return None


def _fit_isotonic_from_db(horizon: str, window_days: int, min_points: int) -> List[float] | None:
    if get_conn is None or IsotonicRegression is None:
        return None
    # Fetch recent predictions matched with realized correctness
    rows: list[tuple[float, bool]] = []
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.proba_up, o.direction_correct
                    FROM predictions p
                    JOIN prediction_outcomes o USING (run_id, horizon)
                    WHERE p.horizon=%s
                      AND o.direction_correct IS NOT NULL
                      AND p.created_at >= now() - (%s || ' days')::interval
                    ORDER BY p.created_at DESC
                    LIMIT 5000
                    """,
                    (horizon, str(window_days)),
                )
                for pr, dc in cur.fetchall() or []:
                    try:
                        rows.append((float(pr), bool(dc)))
                    except Exception:
                        continue
    except Exception as e:  # noqa: BLE001
        logger.debug(f"isotonic: DB fetch failed ({horizon}): {e}")
        return None
    if len(rows) < max(10, int(min_points)):
        return None
    try:
        xs = [max(0.0, min(1.0, r[0])) for r in rows]
        ys = [1.0 if r[1] else 0.0 for r in rows]
        # Fit isotonic regression (monotonic calibration)
        ir = IsotonicRegression(out_of_bounds="clip")  # type: ignore[operator]
        ir.fit(xs, ys)  # type: ignore[call-arg]
        # Export mapping to 0..1 with 0.01 step for lightweight runtime use
        grid = [i / 100.0 for i in range(0, 101)]
        mapped = [float(ir.predict([g])[0]) for g in grid]  # type: ignore[attr-defined]
        return mapped
    except Exception as e:  # noqa: BLE001
        logger.debug(f"isotonic: fit failed ({horizon}): {e}")
        return None


def _load_or_build_iso_map(horizon: str) -> List[float] | None:
    ttl = int(os.getenv("CALIBRATION_TTL_SEC", "21600"))
    win = int(os.getenv("CALIBRATION_WINDOW_DAYS", "30"))
    min_pts = int(os.getenv("CALIBRATION_MIN_POINTS", "50"))
    key = f"calib:isotonic:{horizon}"
    # Try Redis cache
    r = _redis_client()
    if r is not None:
        try:
            raw = r.get(key)
            if raw:
                parts = [p for p in raw.split(",") if p]
                if len(parts) == 101:
                    return [float(x) for x in parts]
        except Exception:
            pass
    # Build from DB
    mapped = _fit_isotonic_from_db(horizon, win, min_pts)
    if mapped and r is not None:
        try:
            r.setex(key, ttl, ",".join(f"{v:.6f}" for v in mapped))
        except Exception:
            pass
    return mapped


def _apply_iso_map(p: float, mapped: List[float]) -> float:
    # mapped is len=101 for grid 0.00..1.00. Do linear interp between bins
    x = max(0.0, min(1.0, float(p)))
    idx = int(x * 100)
    if idx >= 100:
        return float(mapped[100])
    x0 = idx / 100.0
    x1 = (idx + 1) / 100.0
    y0 = float(mapped[idx])
    y1 = float(mapped[idx + 1])
    if x1 <= x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return float(y0 * (1.0 - t) + y1 * t)


def calibrate_proba(
    proba_up: float,
    interval: Tuple[float, float],
    price: float,
    atr: float | None,
    horizon: str | None,
) -> float:
    """General probability calibration.

    - If ENABLE_ISOTONIC_CALIBRATION=1 and enough outcomes exist, apply isotonic mapping for the given horizon.
    - Then apply uncertainty-based compression around 0.5 (interval / ATR aware).
    """
    p = float(max(0.0, min(1.0, proba_up)))
    if os.getenv("ENABLE_ISOTONIC_CALIBRATION", "0") in {"1", "true", "True"} and horizon:
        try:
            m = _load_or_build_iso_map(str(horizon))
            if m:
                p = _apply_iso_map(p, m)
        except Exception:
            pass
    # Always apply light uncertainty-aware compression
    return calibrate_proba_by_uncertainty(p, interval, price, atr)
