from __future__ import annotations

"""Простая детекция аномалий потоков данных по агрегированным метрикам."""

from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class AnomalyResult:
    source_name: str
    metric: str
    value: float
    z_score: float


class StreamAnomalyDetector:
    """Следит за окнами латентности/ошибок и отмечает выбросы."""

    def __init__(self, z_threshold: float = 2.5) -> None:
        self._threshold = z_threshold
        self._stats: Dict[str, Dict[str, float]] = {}

    def detect(self, source: object) -> bool:
        meta = getattr(source, "metadata", {}) or {}
        latency = float(meta.get("latency_hours", 0.0))
        error_rate = float(meta.get("error_rate", 0.0))
        name = getattr(source, "name", "unknown")
        stats = self._stats.setdefault(name, {"lat_mean": latency, "lat_std": 0.1, "err_mean": error_rate, "err_std": 0.05})
        z_latency = self._z(latency, stats["lat_mean"], stats["lat_std"])
        z_error = self._z(error_rate, stats["err_mean"], stats["err_std"])
        is_anomaly = max(z_latency, z_error) >= self._threshold
        # экспоненциальное сглаживание
        self._stats[name]["lat_mean"] = 0.8 * stats["lat_mean"] + 0.2 * latency
        self._stats[name]["lat_std"] = max(0.05, 0.8 * stats["lat_std"] + 0.2 * abs(latency - self._stats[name]["lat_mean"]))
        self._stats[name]["err_mean"] = 0.8 * stats["err_mean"] + 0.2 * error_rate
        self._stats[name]["err_std"] = max(0.02, 0.8 * stats["err_std"] + 0.2 * abs(error_rate - self._stats[name]["err_mean"]))
        return is_anomaly

    def _z(self, value: float, mean: float, std: float) -> float:
        denom = max(std, 1e-6)
        return abs(value - mean) / denom

