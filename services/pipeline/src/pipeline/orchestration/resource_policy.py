from __future__ import annotations

"""Политика распределения ресурсов для MasterOrchestratorAgent."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence

import os


@dataclass(slots=True)
class ResourcePlan:
    run_id: str
    slot: str
    mode: str
    jobs: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    safe_mode: bool = False
    risk_overrides: Dict[str, float] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)

    @property
    def run_master(self) -> bool:
        return "master_flow" in self.jobs

    @property
    def refresh_memory(self) -> bool:
        return "memory_guardian_refresh" in self.jobs

    @property
    def run_red_team(self) -> bool:
        return "red_team_simulation" in self.jobs


class ResourcePolicy:
    """Вычисляет план действий на основе событий и режима рынка."""

    def __init__(self, high_alert_hours: int = 24) -> None:
        self._high_alert_hours = high_alert_hours

    def build_plan(
        self,
        events: Iterable[dict],
        slot: str,
        force_mode: str | None = None,
        pending_events: Sequence[dict] | None = None,
    ) -> ResourcePlan:
        now = datetime.now(timezone.utc)
        mode = force_mode or "Baseline"
        comments: List[str] = []
        jobs = ["strategic_data"]
        pending = list(pending_events or [])
        triggers = [str(evt.get("event_type")) for evt in pending if evt.get("event_type")]
        if triggers:
            comments.extend([f"Событие: {name}" for name in triggers])
        high_impact = self._has_high_impact(events, now)
        if force_mode is None and (high_impact or any(t in {"data_anomaly", "safe_mode"} for t in triggers)):
            mode = "High-Alert"
            comments.append("Высоковажное событие в горизонте")
        if mode == "High-Alert":
            jobs.extend(["master_flow", "memory_guardian_refresh", "red_team_simulation"])
        else:
            jobs.append("master_flow")
            if now.hour % 6 == 0:
                jobs.append("memory_guardian_refresh")
        safe_mode = mode == "High-Alert"
        risk_overrides: Dict[str, float] = {}
        if safe_mode:
            risk_overrides["risk_per_trade"] = float(os.getenv("SAFE_MODE_RISK_PER_TRADE", "0.003"))
            risk_overrides["leverage_cap"] = float(os.getenv("SAFE_MODE_LEVERAGE_CAP", "5"))
            comments.append("Safe-mode активирован: снижено плечо и риск на сделку")
        plan = ResourcePlan(
            run_id=now.strftime("%Y%m%dT%H%M%S"),
            slot=slot,
            mode=mode,
            jobs=list(dict.fromkeys(jobs)),
            comments=comments,
            safe_mode=safe_mode,
            risk_overrides=risk_overrides,
            triggers=triggers,
        )
        return plan

    def _has_high_impact(self, events: Iterable[dict], now: datetime) -> bool:
        horizon = self._high_alert_hours
        for event in events:
            importance = str(event.get("importance", "low")).lower()
            try:
                ts = datetime.fromisoformat(str(event.get("start_time")))
            except Exception:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            delta_h = (ts - now).total_seconds() / 3600.0
            if 0.0 <= delta_h <= horizon and importance in {"high", "medium"}:
                return True
        return False
