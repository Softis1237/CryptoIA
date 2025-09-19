from __future__ import annotations

"""Подсистема динамической оценки доверия к источникам данных."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from loguru import logger

from ...infra import db


@dataclass(slots=True)
class ExistingSource:
    """Запись из БД с текущим рейтингом доверия."""

    name: str
    trust_score: float
    provider: str
    url: str
    tags: List[str]


@dataclass(slots=True)
class TrustUpdate:
    """Результат перерасчёта доверия."""

    source_name: str
    previous_score: float
    new_score: float
    reason: str
    should_escalate: bool
    meta: Dict[str, float]

    def delta(self) -> float:
        return self.new_score - self.previous_score

    def explain(self) -> str:
        return (
            f"Источник {self.source_name}: рейтинг {self.previous_score:.2f} → {self.new_score:.2f}. "
            f"Причина: {self.reason}."
        )


class TrustMonitor:
    """Высчитывает доверие на основе корреляций и кросс-валидации."""

    def __init__(self, min_trust: float = 0.1, max_trust: float = 0.98) -> None:
        self._min_trust = min_trust
        self._max_trust = max_trust

    def fetch_existing(self) -> List[ExistingSource]:
        rows = db.fetch_data_sources(limit=200)
        existing: List[ExistingSource] = []
        for row in rows:
            existing.append(
                ExistingSource(
                    name=row["name"],
                    trust_score=float(row.get("trust_score", 0.5)),
                    provider=row.get("provider", "unknown"),
                    url=row.get("url", ""),
                    tags=list(row.get("tags", [])),
                )
            )
        return existing

    def recalculate(self, candidates: Iterable[object]) -> List[TrustUpdate]:
        existing = {src.name: src for src in self.fetch_existing()}
        updates: List[TrustUpdate] = []
        for item in candidates:
            try:
                name = getattr(item, "name")
                base_trust = float(getattr(item, "reputation", 0.5))
                popularity = float(getattr(item, "popularity", base_trust))
                corr = float((getattr(item, "metadata", {}) or {}).get("corr", 0.0))
            except Exception as exc:  # noqa: BLE001
                logger.warning("TrustMonitor: некорректный кандидат %s: %s", item, exc)
                continue
            prev = existing.get(name)
            prev_score = prev.trust_score if prev else 0.3
            new_score = self._clip(self._aggregate(base_trust, popularity, corr))
            if abs(new_score - prev_score) < 0.01 and prev is not None:
                continue
            reason = "new_source" if prev is None else "score_adjusted"
            should_escalate = new_score <= 0.35 or corr < -0.2
            updates.append(
                TrustUpdate(
                    source_name=name,
                    previous_score=prev_score,
                    new_score=new_score,
                    reason=reason,
                    should_escalate=should_escalate,
                    meta={"corr": corr, "popularity": popularity},
                )
            )
        return updates

    def _aggregate(self, base: float, popularity: float, corr: float) -> float:
        score = 0.6 * base + 0.3 * popularity + 0.1 * max(-1.0, min(1.0, corr))
        return score

    def _clip(self, value: float) -> float:
        return max(self._min_trust, min(self._max_trust, value))
