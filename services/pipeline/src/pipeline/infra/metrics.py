from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict

from loguru import logger

try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
except Exception:  # noqa: BLE001
    CollectorRegistry = None  # type: ignore[assignment]
    Gauge = None  # type: ignore[assignment]
    push_to_gateway = None  # type: ignore[assignment]


@contextmanager
def timed(durations: Dict[str, float], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        durations[key] = durations.get(key, 0.0) + (time.perf_counter() - t0)


def push_durations(job: str, durations: Dict[str, float], labels: Dict[str, str] | None = None) -> None:
    gateway = os.getenv("PROM_PUSHGATEWAY_URL")
    if not gateway or CollectorRegistry is None:
        return
    labels = labels or {}
    try:
        registry = CollectorRegistry()
        g = Gauge("pipeline_step_seconds", "Duration of pipeline step", ["step", *labels.keys()], registry=registry)
        for k, v in durations.items():
            g.labels(step=k, **labels).set(v)
        push_to_gateway(gateway, job=job, registry=registry)
        logger.info(f"Pushed metrics to {gateway} job={job}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Prometheus push failed: {e}")


def push_values(job: str, values: Dict[str, float], labels: Dict[str, str] | None = None) -> None:
    gateway = os.getenv("PROM_PUSHGATEWAY_URL")
    if not gateway or CollectorRegistry is None:
        return
    labels = labels or {}
    try:
        registry = CollectorRegistry()
        g = Gauge("pipeline_value", "Arbitrary pipeline numeric value", ["name", *labels.keys()], registry=registry)
        for k, v in values.items():
            g.labels(name=k, **labels).set(v)
        push_to_gateway(gateway, job=job, registry=registry)
        logger.info(f"Pushed values to {gateway} job={job}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Prometheus push (values) failed: {e}")
