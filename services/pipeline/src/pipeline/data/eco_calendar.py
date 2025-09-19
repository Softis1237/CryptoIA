from __future__ import annotations

"""Экономический календарь (локальный суррогат без сетевых запросов)."""

from datetime import datetime, timedelta, timezone
from typing import Iterable, List


def fetch_upcoming_events(horizon_hours: int = 48) -> List[dict]:
    """Вернуть список событий с полями title, importance, start_time."""

    now = datetime.now(timezone.utc)
    base_events = [
        {
            "title": "FOMC Minutes",
            "importance": "high",
            "start_time": (now + timedelta(hours=12)).isoformat(),
            "category": "macro",
        },
        {
            "title": "US CPI YoY",
            "importance": "high",
            "start_time": (now + timedelta(hours=30)).isoformat(),
            "category": "inflation",
        },
        {
            "title": "ECB Press Conference",
            "importance": "medium",
            "start_time": (now + timedelta(hours=40)).isoformat(),
            "category": "central_bank",
        },
        {
            "title": "Crypto Options Expiry",
            "importance": "medium",
            "start_time": (now + timedelta(hours=55)).isoformat(),
            "category": "crypto",
        },
    ]
    out: List[dict] = []
    for event in base_events:
        ts = datetime.fromisoformat(event["start_time"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta_h = (ts - now).total_seconds() / 3600.0
        if 0 <= delta_h <= horizon_hours:
            out.append(event)
    return out

