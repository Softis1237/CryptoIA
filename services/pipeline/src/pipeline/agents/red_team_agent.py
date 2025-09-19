from __future__ import annotations

"""Red Team Agent — создаёт стресс-сценарии на основе прошлых ошибок."""

import os
from dataclasses import replace
from typing import Any, Dict, List

from pydantic import BaseModel

from ..infra.metrics import push_values
from ..simulations.red_team.scenario import build_scenario_from_lesson, RedTeamScenario
from ..trading.backtest.runner import run_from_dataframe
from .base import AgentResult, BaseAgent
from .memory_guardian_agent import MemoryGuardianAgent


class RedTeamPayload(BaseModel):
    run_id: str
    symbol: str = "BTCUSDT"
    scenarios: int = 5
class RedTeamAgent(BaseAgent):
    name = "red-team-agent"
    priority = 70

    def __init__(self, guardian: MemoryGuardianAgent | None = None) -> None:
        self._guardian = guardian or MemoryGuardianAgent()

    def run(self, payload: dict) -> AgentResult:
        params = RedTeamPayload(**payload)
        lessons = self._guardian.lessons_for_red_team(scope="trading", limit=params.scenarios)
        scenario_objs = self._build_scenarios(lessons, params)
        runs = self._execute_backtests(scenario_objs, params)
        push_values(
            job="red_team_agent",
            values={"red_team_scenarios": float(len(scenario_objs))},
            labels={"symbol": params.symbol},
        )
        scenarios_summary = []
        for run in runs:
            sc = run["scenario"]
            summary = {
                "name": sc.name,
                "description": sc.description,
                "parameters": sc.parameters,
                "hypothesis": sc.description,
                "metrics": run.get("metrics", {}),
                "error": run.get("error"),
            }
            scenarios_summary.append(summary)
        suggestions = self._analyze(runs)
        return AgentResult(
            name=self.name,
            ok=True,
            output={"scenarios": scenarios_summary, "suggestions": suggestions},
        )

    def _build_scenarios(self, lessons: List[dict], params: RedTeamPayload) -> List[RedTeamScenario]:
        scenarios: List[RedTeamScenario] = []
        strategy_path = os.getenv(
            "RED_TEAM_STRATEGY",
            "pipeline.trading.backtest.strategies:MovingAverageCrossStrategy",
        )
        base_price = float(os.getenv("RED_TEAM_BASE_PRICE", "30000"))
        for lesson in lessons:
            payload = lesson.get("lesson") or {}
            error_type = str(payload.get("error_type", "unknown"))
            regime = str(payload.get("market_regime", "range"))
            after = float(lesson.get("outcome_after", payload.get("outcome_after", 0.0)) or 0.0)
            confidence = float(lesson.get("confidence_before", payload.get("confidence_before", 0.5)) or 0.5)
            stress = max(1.0, 1.5 + abs(after) * 5.0)
            hypothesis = f"Если повторится {error_type} при режиме {regime}, уменьшить позицию на {stress*10:.0f}%"
            recommendation = payload.get("correct_action_suggestion") or "Перепроверить сигналы"
            payload.setdefault("symbol", params.symbol)
            payload.setdefault("planned_bias", "bullish" if after >= 0 else "bearish")
            scenario = build_scenario_from_lesson(
                payload,
                stress_factor=round(stress, 2),
                strategy_path=strategy_path,
                base_price=base_price,
            )
            scenario.parameters.update(
                {
                    "confidence_before": confidence,
                    "outcome_after": after,
                    "recommendation": recommendation,
                    "hypothesis": hypothesis,
                }
            )
            scenarios.append(scenario)
        return scenarios

    def _execute_backtests(
        self, scenarios: List[RedTeamScenario], params: RedTeamPayload
    ) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        for scenario in scenarios:
            try:
                cfg = replace(
                    scenario.config,
                    metadata={**scenario.config.metadata, "scenario": scenario.name},
                )
                report = run_from_dataframe(
                    scenario.data.copy(),
                    config=cfg,
                    strategy_path=scenario.strategy_path,
                )
                runs.append(
                    {
                        "scenario": scenario,
                        "report": report,
                        "metrics": report.metrics,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                runs.append({"scenario": scenario, "error": str(exc)})
        return runs

    def _analyze(self, runs: List[Dict[str, Any]]) -> List[str]:
        suggestions: List[str] = []
        for run in runs:
            scenario: RedTeamScenario = run["scenario"]
            if run.get("error"):
                suggestions.append(
                    f"{scenario.name}: симуляция не выполнена ({run['error']})"
                )
                continue
            metrics = run.get("metrics", {})
            total_return = metrics.get("total_return", 0.0)
            win_rate = metrics.get("win_rate", 0.0)
            if total_return < 0:
                suggestions.append(
                    f"{scenario.name}: убыток {total_return:.2%}. {scenario.description or 'Пересмотреть правила входа'}"
                )
            elif win_rate < 0.45:
                suggestions.append(
                    f"{scenario.name}: слабый win_rate {win_rate:.0%}. {scenario.description or 'Добавить фильтры'}"
                )
        return suggestions


def main() -> None:
    agent = RedTeamAgent()
    print(agent.run({"run_id": "demo"}).output)


if __name__ == "__main__":
    main()
