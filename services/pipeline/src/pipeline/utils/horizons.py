from __future__ import annotations

"""Utility helpers for forecast horizon handling."""

from datetime import timedelta
from typing import Iterable, List


_SPEC = {
    "15m": {"minutes": 15, "hours": 0.25},
    "30m": {"minutes": 30, "hours": 0.5},
    "1h": {"minutes": 60, "hours": 1.0},
    "2h": {"minutes": 120, "hours": 2.0},
    "4h": {"minutes": 240, "hours": 4.0},
    "12h": {"minutes": 720, "hours": 12.0},
    "24h": {"minutes": 1440, "hours": 24.0},
}


def known_horizons() -> List[str]:
    return list(_SPEC.keys())


def default_forecast_horizons() -> List[str]:
    """Return default ordering of horizons for forecasting."""

    return ["30m", "4h", "12h", "24h"]


def horizon_to_minutes(hz: str) -> int:
    hz = str(hz).lower().strip()
    if hz in _SPEC:
        return int(_SPEC[hz]["minutes"])
    if hz.endswith("m"):
        return int(float(hz[:-1]))
    if hz.endswith("h"):
        return int(float(hz[:-1]) * 60)
    raise ValueError(f"Unknown horizon: {hz}")


def horizon_to_timedelta(hz: str) -> timedelta:
    return timedelta(minutes=horizon_to_minutes(hz))


def horizon_to_hours(hz: str) -> float:
    return horizon_to_minutes(hz) / 60.0


def normalize_horizons(horizons: Iterable[str] | None) -> List[str]:
    if not horizons:
        return default_forecast_horizons()
    seen = []
    for hz in horizons:
        hz_norm = str(hz).lower().strip()
        if hz_norm not in seen:
            seen.append(hz_norm)
    return seen


def minutes_to_horizon(minutes: int) -> str:
    minutes = int(round(minutes))
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    if minutes % 60 == 0:
        hours = minutes // 60
        return "1h" if hours == 1 else f"{hours}h"
    return f"{minutes}m"
