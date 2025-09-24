from __future__ import annotations

"""Подсистема динамической оценки доверия к источникам данных."""

import csv
import os
from dataclasses import dataclass
from pathlib import Path
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


def _normalize_agent_name(agent: str) -> str:
    name = agent.lower()
    if "ensemble" in name or "ml" in name or "model" in name:
        return "models"
    if "vision" in name or "chart" in name:
        return "vision"
    if "smc" in name:
        return "smc"
    if "debate" in name or "arbiter" in name:
        return "debate"
    return name


class TrustMonitor:
    """Высчитывает доверие на основе корреляций и кросс-валидации."""

    def __init__(
        self,
        min_trust: float = 0.1,
        max_trust: float = 0.98,
        calibration_path: str | None = None,
    ) -> None:
        self._min_trust = min_trust
        self._max_trust = max_trust
        self._calibration = self._load_calibration(calibration_path)

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
            cal = self._calibration.get(name.lower())
            if cal:
                try:
                    corr = float(cal.get("signal_correlation", corr))
                except Exception:
                    pass
                try:
                    hist_pop = float(cal.get("popularity", popularity))
                    popularity = (popularity + hist_pop) / 2.0
                except Exception:
                    pass
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
                    meta={
                        "corr": corr,
                        "popularity": popularity,
                        "calibrated": bool(cal),
                    },
                )
            )
        return updates

    def derive_weights_from_lessons(
        self, lessons: Iterable[dict], regime: str | None = None
    ) -> Dict[str, float]:
        """Compute model weight multipliers from structured lessons."""

        adjustments: Dict[str, float] = {}
        for lesson in lessons or []:
            payload = lesson.get("lesson") if isinstance(lesson, dict) else None
            if not isinstance(payload, dict):
                continue
            lesson_regime = str(payload.get("market_regime") or "")
            if regime and lesson_regime and lesson_regime != regime:
                continue
            outcome = float(
                payload.get("outcome_after")
                or lesson.get("outcome_after")
                or 0.0
            )
            agents = payload.get("involved_agents") or []
            if not agents:
                continue
            delta = 0.05 * max(0.5, min(1.5, abs(outcome) or 1.0))
            for agent in agents:
                key = _normalize_agent_name(str(agent))
                current = adjustments.get(key, 1.0)
                if outcome < 0:
                    current -= delta
                elif outcome > 0:
                    current += delta * 0.6
                adjustments[key] = current

        return {k: max(0.5, min(1.3, v)) for k, v in adjustments.items()}

    def _aggregate(self, base: float, popularity: float, corr: float) -> float:
        score = 0.6 * base + 0.3 * popularity + 0.1 * max(-1.0, min(1.0, corr))
        return score

    def _clip(self, value: float) -> float:
        return max(self._min_trust, min(self._max_trust, value))

    def _load_calibration(self, calibration_path: str | None) -> Dict[str, Dict[str, float]]:
        path = Path(calibration_path or os.getenv("SOURCE_TRUST_CALIBRATION", "data/source_trust_calibration.csv"))
        mapping: Dict[str, Dict[str, float]] = {}
        if not path.exists():
            return mapping
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("source") or "").strip().lower()
                    if not name:
                        continue
                    mapping[name] = {
                        "signal_correlation": float(row.get("signal_correlation", 0.0) or 0.0),
                        "popularity": float(row.get("popularity", 0.0) or 0.0),
                    }
        except Exception as exc:  # noqa: BLE001
            logger.warning("TrustMonitor: failed to load calibration %s: %s", path, exc)
        return mapping
