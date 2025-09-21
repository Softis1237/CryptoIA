from __future__ import annotations

"""Агент стратегического управления источниками данных."""

from dataclasses import dataclass
from typing import Iterable, List

from loguru import logger
from pydantic import BaseModel, Field, HttpUrl

from ..infra import db, metrics
from ..infra.run_lock import slot_lock
from ..ops.project_tasks import ensure_task
from .base import AgentResult, BaseAgent
from .strategic import (
    ExistingSource,
    StreamAnomalyDetector,
    TrustMonitor,
    TrustUpdate,
    crawl_catalogs,
)


class StrategicDataPayload(BaseModel):
    """Параметры запуска агента."""

    run_id: str
    slot: str = "strategic_data"
    keywords: List[str] = Field(default_factory=lambda: ["bitcoin", "liquidity"])
    health_check_only: bool = False


@dataclass(slots=True)
class StrategicSource:
    """Нормализованный источник данных."""

    name: str
    url: HttpUrl
    provider: str
    tags: List[str]
    popularity: float
    reputation: float
    metadata: dict[str, float]

    @property
    def base_trust(self) -> float:
        return max(0.0, min(1.0, 0.5 * self.popularity + 0.5 * self.reputation))


class StrategicDataAgent(BaseAgent):
    """Управляет полным жизненным циклом внешних источников информации."""

    name = "strategic-data-agent"
    priority = 35

    def __init__(
        self,
        trust_monitor: TrustMonitor | None = None,
        anomaly_detector: StreamAnomalyDetector | None = None,
    ) -> None:
        self._trust = trust_monitor or TrustMonitor()
        self._anomaly = anomaly_detector or StreamAnomalyDetector()

    def run(self, payload: dict) -> AgentResult:
        params = StrategicDataPayload(**payload)
        with slot_lock(params.slot):
            if params.health_check_only:
                existing = self._trust.fetch_existing()
                return AgentResult(
                    name=self.name,
                    ok=True,
                    output={"existing_sources": [src.name for src in existing]},
                )

            existing_registry = {src.name: src for src in self._trust.fetch_existing()}
            discoveries = list(self._discover(params))
            updates = self._trust.recalculate(discoveries)
            anomalies = self._persist_and_check(discoveries, updates, params, existing_registry)
            self._maybe_escalate(updates, anomalies)
        return AgentResult(
            name=self.name,
            ok=True,
            output={
                "discovered": [src.name for src in discoveries],
                "updated": len(updates),
                "anomalies": anomalies,
            },
        )

    def _discover(self, params: StrategicDataPayload) -> Iterable[StrategicSource]:
        for candidate in crawl_catalogs(params.keywords):
            yield StrategicSource(
                name=candidate.name,
                url=HttpUrl(str(candidate.url)),
                provider=candidate.provider,
                tags=list(candidate.tags),
                popularity=float(candidate.popularity),
                reputation=float(candidate.reputation),
                metadata=dict(candidate.metadata),
            )

    def _persist_and_check(
        self,
        discoveries: Iterable[StrategicSource],
        updates: List[TrustUpdate],
        params: StrategicDataPayload,
        existing_registry: dict[str, ExistingSource],
    ) -> List[str]:
        anomalies: List[str] = []
        trust_map = {upd.source_name: upd.new_score for upd in updates}
        for src in discoveries:
            trust_score = trust_map.get(src.name, src.base_trust)
            db.upsert_data_source(
                name=src.name,
                url=str(src.url),
                provider=src.provider,
                tags=src.tags,
                popularity=src.popularity,
                reputation=src.reputation,
                trust_score=trust_score,
                metadata=src.metadata,
            )
            if src.name not in existing_registry:
                ensure_task(
                    summary=f"[Data] Встроить источник {src.name}",
                    description=(
                        f"Найден новый источник данных {src.url} (provider={src.provider}). "
                        f"popularity={src.popularity:.2f}, reputation={src.reputation:.2f}. "
                        "Проверьте схему данных, авторизацию и лимиты; добавьте мониторинг доверия."
                    ),
                    priority="medium",
                    tags=["data_source", "ingest"],
                )
                existing_registry[src.name] = ExistingSource(
                    name=src.name,
                    trust_score=trust_score,
                    provider=src.provider,
                    url=str(src.url),
                    tags=src.tags,
                )
            if self._anomaly.detect(src):
                anomalies.append(src.name)
                db.insert_data_source_anomaly(
                    source_name=src.name,
                    run_id=params.run_id,
                    meta={"latency_hours": src.metadata.get("latency_hours", 0.0)},
                )
        for upd in updates:
            db.insert_data_source_history(
                source_name=upd.source_name,
                delta=upd.delta(),
                new_score=upd.new_score,
                reason=upd.reason,
                meta=upd.meta,
            )
        return anomalies

    def _maybe_escalate(self, updates: Iterable[TrustUpdate], anomalies: List[str]) -> None:
        for upd in updates:
            if upd.should_escalate:
                ensure_task(
                    summary=f"[Data] Проверить источник {upd.source_name}",
                    description=upd.explain(),
                    priority="high",
                    tags=["data_quality"],
                )
        for name in anomalies:
            ensure_task(
                summary=f"[Data] Аномалия потока {name}",
                description=(
                    "Detector зафиксировал превышение латентности/ошибок. "
                    f"run_id={params.run_id}. Проверьте зеркальные источники и переключите safe-mode."
                ),
                priority="critical",
                tags=["anomaly", "safe_mode"],
            )
        if anomalies:
            metrics.push_values(
                job="strategic_data_agent",
                values={"data_source_anomaly": float(len(anomalies))},
            )
            try:
                db.insert_orchestrator_event(
                    "data_anomaly",
                    {"sources": anomalies, "run_id": params.run_id},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to enqueue orchestrator event for anomalies: %s", exc)


def main() -> None:
    agent = StrategicDataAgent()
    result = agent.run({"run_id": "demo", "keywords": ["btc", "orderflow"]})
    print(result.output)


if __name__ == "__main__":
    main()
