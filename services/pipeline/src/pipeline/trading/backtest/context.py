from __future__ import annotations

"""
HistoricalContextProvider — простой провайдер исторического контекста для бэктестов.

Идея: по временной метке вернуть фрагменты новостей/ончейна/соц‑метрик, которые были бы доступны в тот момент.

Для MVP поддерживает инъекцию заранее подготовленных списков: news (list[dict]), onchain (list[dict])
с полем 'ts' (ISO8601 или unix seconds). Возвращает последние записи за окна 6–12 часов.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def _to_ts(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc)
    try:
        # unix seconds
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        pass
    return pd.to_datetime(v, utc=True).to_pydatetime()


@dataclass
class HistoricalContextProvider:
    news: Optional[Iterable[Dict[str, Any]]] = None
    onchain: Optional[Iterable[Dict[str, Any]]] = None
    window_hours_news: int = 12
    window_hours_onchain: int = 24

    def at(self, ts: datetime) -> Dict[str, Any]:
        ref = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        out: Dict[str, Any] = {}
        if self.news:
            win = ref - timedelta(hours=max(1, int(self.window_hours_news)))
            items = []
            for n in self.news:
                t = _to_ts(n.get("ts")) if n else None
                if t and win <= t <= ref:
                    items.append(n)
            out["news"] = items[:20]
        if self.onchain:
            win = ref - timedelta(hours=max(1, int(self.window_hours_onchain)))
            items = []
            for o in self.onchain:
                t = _to_ts(o.get("ts")) if o else None
                if t and win <= t <= ref:
                    items.append(o)
            out["onchain"] = items[:20]
        return out

