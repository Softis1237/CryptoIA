from __future__ import annotations

"""Дополнительные адаптеры для реальных источников данных."""

import contextlib
import csv
from dataclasses import replace
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, Iterable, Protocol

try:  # pragma: no cover - сеть опциональна
    import requests
except Exception:  # noqa: BLE001
    requests = None  # type: ignore

from .discovery import DiscoveryCandidate


class ProviderAdapter(Protocol):
    """Интерфейс адаптера источника."""

    def enrich(self, candidate: DiscoveryCandidate) -> DiscoveryCandidate:  # pragma: no cover - протокол
        ...


def _to_hours(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - parsed
        return max(0.0, delta.total_seconds() / 3600.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _correlation(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_list = list(xs)
    y_list = list(ys)
    n = min(len(x_list), len(y_list))
    if n < 2:
        return 0.0
    x_list = x_list[-n:]
    y_list = y_list[-n:]
    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_list, y_list))
    den_x = sum((x - mean_x) ** 2 for x in x_list)
    den_y = sum((y - mean_y) ** 2 for y in y_list)
    denom = (den_x * den_y) ** 0.5
    if denom == 0:
        return 0.0
    return num / denom


class CoinGeckoAdapter:
    url = "https://api.coingecko.com/api/v3/coins/bitcoin"

    def enrich(self, candidate: DiscoveryCandidate) -> DiscoveryCandidate:
        if requests is None:
            return candidate
        meta: Dict[str, float] = dict(candidate.metadata)
        try:
            r = requests.get(self.url, timeout=5)
            r.raise_for_status()
            data = r.json()
            last_updated = data.get("market_data", {}).get("last_updated")
            meta["latency_hours"] = _to_hours(last_updated)
            market_cap = _safe_float(data.get("market_data", {}).get("market_cap", {}).get("usd"))
            volume = _safe_float(data.get("market_data", {}).get("total_volume", {}).get("usd"))
            popularity = min(1.0, candidate.popularity + (market_cap / 1e11)) if market_cap else candidate.popularity
            reputation = min(1.0, candidate.reputation + (volume / 1e10)) if volume else candidate.reputation
            meta["market_cap_usd"] = market_cap
            meta["volume_usd"] = volume
            meta["error_rate"] = 0.0
            chart = requests.get(
                "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": 14},
                timeout=5,
            ).json()
            prices = [pt[1] for pt in chart.get("prices", [])]
            vol_series = [pt[1] for pt in chart.get("total_volumes", [])]
            meta["corr"] = float(_correlation(prices, vol_series))
        except Exception:  # noqa: BLE001
            meta.setdefault("latency_hours", 0.0)
            meta["error_rate"] = 0.05
            popularity = candidate.popularity * 0.95
            reputation = candidate.reputation * 0.95
        else:
            return replace(candidate, popularity=popularity, reputation=reputation, metadata=meta)
        return replace(candidate, metadata=meta, popularity=popularity, reputation=reputation)


class CryptoDataDownloadAdapter:
    def enrich(self, candidate: DiscoveryCandidate) -> DiscoveryCandidate:
        if requests is None:
            return candidate
        meta: Dict[str, float] = dict(candidate.metadata)
        try:
            r = requests.get(str(candidate.url), timeout=5)
            r.raise_for_status()
            lines = r.text.strip().splitlines()
            # пропускаем заголовок, ищем последнюю дату
            last_line = next((line for line in lines if not line.startswith("#")), "")
            parts = last_line.split(",")
            last_ts = parts[0] if parts else ""
            meta["latency_hours"] = _to_hours(last_ts)
            meta["rows"] = float(len(lines))
            meta["error_rate"] = 0.0
            reader = csv.DictReader(StringIO("\n".join(lines)))
            closes = []
            volumes = []
            for row in reader:
                closes.append(_safe_float(row.get("close")))
                volumes.append(_safe_float(row.get("Volume USDT")))
            meta["corr"] = float(_correlation(closes, volumes))
        except Exception:  # noqa: BLE001
            meta.setdefault("latency_hours", 12.0)
            meta["error_rate"] = 0.15
        return replace(candidate, metadata=meta)


class GenericHeadAdapter:
    """Фолбэк: считаем только латентность по Last-Modified."""

    def __init__(self, timeout: int = 4) -> None:
        self._timeout = timeout

    def enrich(self, candidate: DiscoveryCandidate) -> DiscoveryCandidate:
        if requests is None:
            return candidate
        meta: Dict[str, float] = dict(candidate.metadata)
        try:
            r = requests.head(str(candidate.url), timeout=self._timeout)
            if r.status_code >= 400:
                r.raise_for_status()
            last_mod = r.headers.get("Last-Modified")
            meta.setdefault("latency_hours", _to_hours(last_mod))
            meta.setdefault("error_rate", 0.05)
        except Exception:  # noqa: BLE001
            meta.setdefault("latency_hours", 24.0)
            meta.setdefault("error_rate", 0.25)
        return replace(candidate, metadata=meta)


class KaggleAdapter(GenericHeadAdapter):
    pass


_ADAPTERS: Dict[str, ProviderAdapter] = {
    "coingecko": CoinGeckoAdapter(),
    "cryptodatadownload": CryptoDataDownloadAdapter(),
    "dune": GenericHeadAdapter(),
    "kaggle": KaggleAdapter(),
}


def enrich_candidate(candidate: DiscoveryCandidate) -> DiscoveryCandidate:
    adapter = _ADAPTERS.get(candidate.provider.lower())
    if adapter is None:
        return candidate
    with contextlib.suppress(Exception):
        return adapter.enrich(candidate)
    return candidate
