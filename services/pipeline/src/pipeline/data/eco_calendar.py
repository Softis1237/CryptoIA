from __future__ import annotations

"""Экономический календарь (локальный суррогат без сетевых запросов)."""

import os
from datetime import datetime, timedelta, timezone
from typing import List

try:  # pragma: no cover - сеть опциональна
    import requests
except Exception:  # noqa: BLE001
    requests = None  # type: ignore


_FALLBACK_EVENTS = [
    {
        "title": "FOMC Minutes",
        "importance": "high",
        "start_time": 12.0,
        "category": "macro",
    },
    {
        "title": "US CPI YoY",
        "importance": "high",
        "start_time": 30.0,
        "category": "inflation",
    },
    {
        "title": "ECB Press Conference",
        "importance": "medium",
        "start_time": 40.0,
        "category": "central_bank",
    },
    {
        "title": "Crypto Options Expiry",
        "importance": "medium",
        "start_time": 55.0,
        "category": "crypto",
    },
]


def _fallback(now: datetime, horizon_hours: int) -> List[dict]:
    out: List[dict] = []
    for item in _FALLBACK_EVENTS:
        ts = now + timedelta(hours=float(item.get("start_time", 0.0)))
        out.append({
            "title": item["title"],
            "importance": item["importance"],
            "start_time": ts.isoformat(),
            "category": item["category"],
        })
    return _filter_by_horizon(out, now, horizon_hours)


def _filter_by_horizon(events: List[dict], now: datetime, horizon_hours: int) -> List[dict]:
    out: List[dict] = []
    for event in events:
        ts_val = event.get("start_time")
        try:
            ts = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta_h = (ts - now).total_seconds() / 3600.0
        if 0 <= delta_h <= horizon_hours:
            out.append(event)
    return out


def fetch_upcoming_events(horizon_hours: int = 48) -> List[dict]:
    """Вернуть список событий календаря, пытаясь взять живые данные."""

    now = datetime.now(timezone.utc)
    if requests is None:
        return _fallback(now, horizon_hours)
    url = os.getenv(
        "ECO_CALENDAR_URL",
        "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json",
    )
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        events: List[dict] = []
        for item in data:
            title = item.get("event", "")
            if not title:
                continue
            timestamp = item.get("timestamp") or item.get("date")
            impact = str(item.get("impact", "low")).lower()
            category = item.get("country", "macro")
            ts_iso = _to_iso(item.get("timestamp") or item.get("date"), now)
            events.append(
                {
                    "title": title,
                    "importance": impact,
                    "start_time": ts_iso,
                    "category": category,
                }
            )
        filtered = _filter_by_horizon(events, now, horizon_hours)
        if filtered:
            return filtered
    except Exception:
        return _fallback(now, horizon_hours)
    return _fallback(now, horizon_hours)
def _to_iso(ts: object, fallback: datetime) -> str:
    try:
        if ts is None:
            raise ValueError
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:  # noqa: BLE001
        return fallback.isoformat()
