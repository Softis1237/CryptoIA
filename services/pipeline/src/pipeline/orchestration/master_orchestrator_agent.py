from __future__ import annotations

"""Master Orchestrator Agent — адаптивный центр управления пайплайном."""

from dataclasses import dataclass
from typing import Dict, Iterable

from loguru import logger
from pydantic import BaseModel, Field

from ..agents.base import AgentResult, BaseAgent
from ..agents.strategic_data_agent import StrategicDataAgent
from ..agents.memory_guardian_agent import MemoryGuardianAgent
from ..agents.red_team_agent import RedTeamAgent
from ..infra.metrics import push_values
from ..infra.run_lock import slot_lock
from ..data.eco_calendar import fetch_upcoming_events
from .resource_policy import ResourcePlan, ResourcePolicy
from ..agents.master import run_master_flow
from ..infra.db import (
    fetch_pending_orchestrator_events,
    mark_orchestrator_events_processed,
)


class OrchestratorPayload(BaseModel):
    slot: str = "orchestrator"
    horizon_hours: int = 48
    force_mode: str | None = None
    keywords: list[str] = Field(default_factory=lambda: ["onchain", "orderflow"])


@dataclass(slots=True)
class OrchestratorContext:
    policy: ResourcePolicy
    memory_agent: MemoryGuardianAgent
    strategic_agent: StrategicDataAgent
    red_team_agent: RedTeamAgent


class MasterOrchestratorAgent(BaseAgent):
    name = "master-orchestrator"
    priority = 1

    def __init__(self, policy: ResourcePolicy | None = None) -> None:
        self._ctx = OrchestratorContext(
            policy=policy or ResourcePolicy(),
            memory_agent=MemoryGuardianAgent(),
            strategic_agent=StrategicDataAgent(),
            red_team_agent=RedTeamAgent(),
        )

    def run(self, payload: dict) -> AgentResult:
        params = OrchestratorPayload(**payload)
        with slot_lock(params.slot):
            events = fetch_upcoming_events(params.horizon_hours)
            pending_events = fetch_pending_orchestrator_events()
            summary_status = "processed"
            force_mode = params.force_mode
            if not force_mode:
                for evt in pending_events:
                    if evt.get("event_type") in {"data_anomaly", "safe_mode", "post_mortem_feedback"}:
                        force_mode = "High-Alert"
                        summary_status = "escalated"
                        break
            plan = self._ctx.policy.build_plan(events, slot=params.slot, force_mode=force_mode)
            summary = self._execute(plan, params, events)
            final_mode = plan.mode
            if summary.get("anomaly_triggered") and plan.mode != "High-Alert":
                alert_plan = self._ctx.policy.build_plan(events, slot=params.slot, force_mode="High-Alert")
                summary["escalated_plan"] = self._execute(alert_plan, params, events, skip_jobs={"strategic_data"})
                final_mode = "High-Alert"
                summary["mode"] = "High-Alert"
            summary["pending_events"] = pending_events
            if pending_events:
                try:
                    mark_orchestrator_events_processed([evt["id"] for evt in pending_events], status=summary_status)
                except Exception:
                    logger.exception("Failed to mark orchestrator events processed")
            push_values(
                job="master_orchestrator",
                values={"orchestrator_mode": 1.0 if final_mode == "High-Alert" else 0.0},
                labels={"slot": params.slot},
            )
            return AgentResult(name=self.name, ok=True, output=summary)

    def _execute(
        self,
        plan: ResourcePlan,
        params: OrchestratorPayload,
        events: Iterable[dict],
        skip_jobs: set[str] | None = None,
    ) -> Dict[str, object]:
        results: Dict[str, object] = {"mode": plan.mode, "jobs": plan.jobs, "events": list(events)}
        skip_jobs = skip_jobs or set()
        for job in plan.jobs:
            if job in skip_jobs:
                continue
            try:
                if job == "strategic_data":
                    out = self._ctx.strategic_agent.run({"run_id": plan.run_id, "slot": "strategic_data", "keywords": params.keywords})
                    results["strategic_data"] = out.output
                    anomalies = out.output.get("anomalies") if isinstance(out.output, dict) else None
                    if anomalies:
                        results["anomaly_triggered"] = True
                        results.setdefault("alerts", {})["strategic_data"] = anomalies
                elif job == "master_flow":
                    results["master_flow"] = run_master_flow(slot=plan.slot)
                elif job == "memory_guardian_refresh":
                    results["memory_refresh"] = self._ctx.memory_agent.refresh(scope="trading")
                elif job == "red_team_simulation":
                    results["red_team"] = self._ctx.red_team_agent.run({"run_id": plan.run_id, "symbol": "BTCUSDT"}).output
            except Exception as exc:  # noqa: BLE001
                logger.exception("MasterOrchestrator job %s failed: %s", job, exc)
                results.setdefault("failed_jobs", []).append({"job": job, "error": str(exc)})
        results["comments"] = plan.comments
        return results


def main() -> None:
    agent = MasterOrchestratorAgent()
    print(agent.run({}).output)


if __name__ == "__main__":
    main()
