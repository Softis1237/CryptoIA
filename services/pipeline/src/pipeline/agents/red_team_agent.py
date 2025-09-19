from __future__ import annotations

"""Red Team Agent — создаёт стресс-сценарии на основе прошлых ошибок."""

from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import BaseModel

from ..infra.metrics import push_values
from .base import AgentResult, BaseAgent
from .memory_guardian_agent import MemoryGuardianAgent


class RedTeamPayload(BaseModel):
    run_id: str
    symbol: str = "BTCUSDT"
    scenarios: int = 5


@dataclass(slots=True)
class Scenario:
    lesson: Dict[str, Any]
    stress_factor: float
    hypothesis: str
    recommendation: str

    def as_dict(self) -> Dict[str, Any]:
        payload = dict(self.lesson)
        payload.update(
            {
                "stress_factor": self.stress_factor,
                "hypothesis": self.hypothesis,
                "recommendation": self.recommendation,
            }
        )
        return payload


class RedTeamAgent(BaseAgent):
    name = "red-team-agent"
    priority = 70

    def __init__(self, guardian: MemoryGuardianAgent | None = None) -> None:
        self._guardian = guardian or MemoryGuardianAgent()

    def run(self, payload: dict) -> AgentResult:
        params = RedTeamPayload(**payload)
        lessons = self._guardian.lessons_for_red_team(scope="trading", limit=params.scenarios)
        scenarios = [sc.as_dict() for sc in self._build_scenarios(lessons, params)]
        push_values(
            job="red_team_agent",
            values={"red_team_scenarios": float(len(scenarios))},
            labels={"symbol": params.symbol},
        )
        suggestions = [sc.get("recommendation") for sc in scenarios]
        return AgentResult(name=self.name, ok=True, output={"scenarios": scenarios, "suggestions": suggestions})

    def _build_scenarios(self, lessons: List[dict], params: RedTeamPayload) -> List[Scenario]:
        scenarios: List[Scenario] = []
        for lesson in lessons:
            payload = lesson.get("lesson") or {}
            error_type = str(payload.get("error_type", "unknown"))
            regime = str(payload.get("market_regime", "range"))
            after = float(lesson.get("outcome_after", payload.get("outcome_after", 0.0)) or 0.0)
            confidence = float(lesson.get("confidence_before", payload.get("confidence_before", 0.5)) or 0.5)
            stress = max(1.0, 1.5 + abs(after) * 5.0)
            hypothesis = f"Если повторится {error_type} при режиме {regime}, уменьшить позицию на {stress*10:.0f}%"
            recommendation = payload.get("correct_action_suggestion") or "Перепроверить сигналы"
            scenarios.append(
                Scenario(
                    lesson=payload,
                    stress_factor=round(stress, 2),
                    hypothesis=hypothesis,
                    recommendation=recommendation,
                )
            )
        return scenarios


def main() -> None:
    agent = RedTeamAgent()
    print(agent.run({"run_id": "demo"}).output)


if __name__ == "__main__":
    main()
