from __future__ import annotations

"""SelfCritiqueAgent — адвокат дьявола для InvestmentArbiter."""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from ..infra import metrics
from ..reasoning.llm import call_openai_json
from .base import AgentResult, BaseAgent


@dataclass(slots=True)
class SelfCritiqueAgent(BaseAgent):
    name: str = "self-critique-agent"
    priority: int = 40

    def __init__(self, llm_call=call_openai_json) -> None:
        self.name = "self-critique-agent"
        self.priority = 40
        self._llm_call = llm_call

    def run(self, payload: dict) -> AgentResult:
        analysis = payload.get("analysis") or {}
        context = payload.get("context") or {}
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(analysis, context)
        model = os.getenv("OPENAI_MODEL_SELFCRIT", os.getenv("OPENAI_MODEL_MASTER"))
        raw = self._llm_call(system_prompt, user_prompt, model=model, temperature=0.1)
        report = self._parse(raw)
        try:
            metrics.push_values(
                job=self.name,
                values={
                    "probability_delta": float(report.get("probability_adjustment", 0.0)),
                },
                labels={"recommendation": report.get("recommendation", "UNKNOWN")},
            )
        except Exception:
            pass
        return AgentResult(name=self.name, ok=True, output=report)

    def _system_prompt(self) -> str:
        return (
            "Ты — независимый риск-аналитик. Твоя задача — оспорить вывод основного аналитика"
            " и указать потенциальные угрозы. Верни JSON строго в формате:\n"
            "{\"counterarguments\":[str,...],\"invalidators\":[str,...],\"missing_factors\":[str,...],"
            "\"probability_adjustment\":float,\"recommendation\":\"CONFIRM|REVISE|REJECT\"}."
        )

    def _user_prompt(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        analysis_json = json.dumps(analysis, ensure_ascii=False)
        context_summary = context.get("text") or ""
        return (
            "Основной анализ:\n"
            f"{analysis_json}\n"
            "Контекст:\n"
            f"{context_summary}\n"
            "Ответь последовательно на вопросы: \n"
            "1) Самый сильный контраргумент.\n"
            "2) Какое событие/уровень немедленно обнулит сценарий?\n"
            "3) Что мог забыть аналитик?\n"
            "4) Не завышена ли вероятность? Если да — на сколько пунктов скорректировать (отрицательное значение снижает вероятность)."
        )

    def _parse(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if not raw or raw.get("status") == "error":
            return {
                "counterarguments": ["LLM недоступен — используем базовые проверки."],
                "invalidators": [],
                "missing_factors": [],
                "probability_adjustment": 0.0,
                "recommendation": "REVISE",
                "raw": raw,
            }
        counter = raw.get("counterarguments") or raw.get("arguments") or []
        invalid = raw.get("invalidators") or raw.get("kill_switch") or []
        missing = raw.get("missing_factors") or raw.get("omissions") or []
        adj = raw.get("probability_adjustment") or raw.get("delta") or 0.0
        reco = raw.get("recommendation") or raw.get("verdict") or "REVISE"
        try:
            adj = float(adj)
        except Exception:
            adj = 0.0
        return {
            "counterarguments": [str(x) for x in counter][:5],
            "invalidators": [str(x) for x in invalid][:5],
            "missing_factors": [str(x) for x in missing][:5],
            "probability_adjustment": adj,
            "recommendation": str(reco).upper(),
            "raw": raw,
        }


def main() -> None:  # pragma: no cover
    agent = SelfCritiqueAgent()
    res = agent.run({
        "analysis": {"scenario": "LONG", "probability_pct": 68, "explanation": "demo"},
        "context": {"text": "Demo context"},
    })
    print(res.output)


if __name__ == "__main__":
    main()
