from __future__ import annotations

"""Модуль поиска и первичной оценки источников данных."""

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from pydantic import HttpUrl


@dataclass(slots=True)
class DiscoveryCandidate:
    """Описание найденного источника, пригодного для сохранения в БД."""

    name: str
    url: HttpUrl
    provider: str
    tags: list[str]
    popularity: float
    reputation: float
    metadata: dict[str, float]

    @property
    def base_trust(self) -> float:
        """Начальный рейтинг доверия на основе популярности и репутации."""

        return max(0.0, min(1.0, 0.5 * self.popularity + 0.5 * self.reputation))


def _catalog_snapshot() -> Sequence[dict[str, object]]:
    """Локальный список кандидатов (замена сетевых запросов в офлайн-режиме).

    Источник данных можно расширять — структура максимально простая, чтобы заменить
    статический массив на загрузку из S3 или внешнего API без изменения интерфейса.
    """

    return (
        {
            "name": "Glassnode Funding Rates",
            "url": "https://studio.glassnode.com/metrics?a=BTC&m=derivatives.FundingRatePerpMajorExchanges",
            "provider": "glassnode",
            "tags": ["onchain", "funding", "btc"],
            "popularity": 0.86,
            "reputation": 0.92,
            "metadata": {"latency_hours": 1.0},
        },
        {
            "name": "Coinalyze CVD Dashboard",
            "url": "https://coinalyze.net/bitcoin/usdt/binance/",
            "provider": "coinalyze",
            "tags": ["cvd", "orderflow", "binance"],
            "popularity": 0.74,
            "reputation": 0.71,
            "metadata": {"latency_hours": 0.1},
        },
        {
            "name": "CryptoQuant Exchange Reserves",
            "url": "https://cryptoquant.com/asset/btc/exchange-flows",
            "provider": "cryptoquant",
            "tags": ["exchange", "flow", "btc"],
            "popularity": 0.81,
            "reputation": 0.88,
            "metadata": {"latency_hours": 0.5},
        },
        {
            "name": "Santiment Social Volume",
            "url": "https://app.santiment.net/labs/social-tool?name=Bitcoin",
            "provider": "santiment",
            "tags": ["social", "sentiment", "btc"],
            "popularity": 0.69,
            "reputation": 0.67,
            "metadata": {"latency_hours": 2.0},
        },
        {
            "name": "Kaiko Market Depth",
            "url": "https://terminal.kaiko.com/markets/depth",
            "provider": "kaiko",
            "tags": ["liquidity", "orderbook", "derivatives"],
            "popularity": 0.77,
            "reputation": 0.83,
            "metadata": {"latency_hours": 0.25},
        },
    )


def _score_match(tags: Sequence[str], keywords: Sequence[str]) -> float:
    """Мягкое совпадение по ключевым словам."""

    tag_set = {t.lower() for t in tags}
    kw_set = {k.lower() for k in keywords}
    if not kw_set:
        return 1.0
    matched = len(tag_set & kw_set)
    if matched == 0:
        return 0.0
    return matched / max(1, len(kw_set))


def crawl_catalogs(keywords: Sequence[str]) -> Iterator[DiscoveryCandidate]:
    """Вернуть кандидатов, отсортированных по релевантности запросу."""

    entries = []
    for item in _catalog_snapshot():
        score = _score_match(item["tags"], keywords)
        if score <= 0.0:
            continue
        base = DiscoveryCandidate(
            name=str(item["name"]),
            url=HttpUrl(str(item["url"])),
            provider=str(item["provider"]),
            tags=[str(t) for t in item.get("tags", [])],
            popularity=float(item["popularity"]),
            reputation=float(item["reputation"]),
            metadata={k: float(v) for k, v in (item.get("metadata") or {}).items()},
        )
        entries.append((score * base.base_trust, base))
    entries.sort(key=lambda pair: pair[0], reverse=True)
    for _, candidate in entries:
        yield candidate

