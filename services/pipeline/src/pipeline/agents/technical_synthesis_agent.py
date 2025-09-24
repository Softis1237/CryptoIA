from __future__ import annotations

"""Technical Synthesis Agent — объединяет TA признаки и визуальные сигналы."""

from dataclasses import dataclass
from typing import Dict

from pydantic import BaseModel, Field

from ..infra import metrics
from ..infra.db import fetch_agent_config
from ..infra.run_lock import slot_lock
from .base import AgentResult, BaseAgent


class TechnicalSynthesisPayload(BaseModel):
    run_id: str
    symbol: str
    slot: str = "technical_synthesis"
    indicators: Dict[str, float] = Field(default_factory=dict)
    smc: Dict[str, object] = Field(default_factory=dict)
    vision: Dict[str, object] = Field(default_factory=dict)
    features_meta: Dict[str, object] = Field(default_factory=dict)


@dataclass(slots=True)
class ComponentScore:
    name: str
    score: float
    rationale: str


class TechnicalSynthesisAgent(BaseAgent):
    name = "technical-synthesis-agent"
    priority = 34

    def __init__(self) -> None:
        cfg = fetch_agent_config("TechnicalSynthesis") or {}
        params = cfg.get("parameters") if isinstance(cfg, dict) else None
        default = {"indicators": 1.0, "smc": 1.0, "vision": 1.0}
        weights = {}
        if isinstance(params, dict):
            weights = {
                key: float(params.get(f"weight_{key}", default.get(key, 1.0)))
                for key in default
            }
        self._weights = {**default, **weights}
        self._record_metrics = params.get("record_metrics", True) if isinstance(params, dict) else True

    def run(self, payload: dict) -> AgentResult:
        params = TechnicalSynthesisPayload(**payload)
        with slot_lock(params.slot):
            scores = self._score_components(params)
            verdict = self._combine(scores)
            if self._record_metrics:
                self._push_metrics(verdict)
        return AgentResult(name=self.name, ok=True, output={"verdict": verdict, "components": [s.__dict__ for s in scores]})

    def _score_components(self, params: TechnicalSynthesisPayload) -> list[ComponentScore]:
        scores: list[ComponentScore] = []
        rsi = float(params.indicators.get("rsi_dynamic", params.indicators.get("rsi", 50.0)))
        bb_width = float(params.indicators.get("bb_width_dynamic", params.indicators.get("bb_width", 0.02)))
        trend_bias = float(params.indicators.get("trend_strength", 0.0))
        indicators_score = 0.0
        rationale = []
        if rsi >= 55:
            indicators_score += 0.6
            rationale.append("RSI поддерживает рост")
        elif rsi <= 45:
            indicators_score -= 0.6
            rationale.append("RSI указывает на слабость")
        if bb_width < 0.03:
            indicators_score -= 0.2
            rationale.append("Сжатие волатильности")
        indicators_score += trend_bias
        scores.append(ComponentScore("indicators", indicators_score * self._weights.get("indicators", 1.0), "; ".join(rationale) or "Нейтрально"))

        smc_status = str(params.smc.get("status", "NO_SETUP"))
        smc_score = 0.0
        if smc_status.endswith("BULLISH_SETUP"):
            smc_score += 0.8
        elif smc_status.endswith("BEARISH_SETUP"):
            smc_score -= 0.8
        scores.append(ComponentScore("smc", smc_score * self._weights.get("smc", 1.0), f"SMC статус: {smc_status}"))

        vision_bias = str(params.vision.get("bias", "neutral"))
        vision_conf = float(params.vision.get("confidence", 0.5))
        bias_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
        vision_score = bias_map.get(vision_bias, 0.0) * vision_conf
        scores.append(ComponentScore("vision", vision_score * self._weights.get("vision", 1.0), f"Chart bias: {vision_bias}"))

        return scores

    def _combine(self, scores: list[ComponentScore]) -> dict:
        total = sum(s.score for s in scores)
        if total > 0.6:
            bias = "BULLISH"
        elif total < -0.6:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
        confidence = min(0.95, max(0.05, abs(total) / 2.0))
        return {"bias": bias, "confidence": round(confidence, 3), "score": round(total, 3)}

    def _push_metrics(self, verdict: dict) -> None:
        try:
            metrics.push_values(
                job=self.name,
                values={
                    "score": float(verdict.get("score", 0.0) or 0.0),
                    "confidence": float(verdict.get("confidence", 0.0) or 0.0),
                },
                labels={"bias": str(verdict.get("bias", "NEUTRAL"))},
            )
        except Exception:
            pass


def main() -> None:
    agent = TechnicalSynthesisAgent()
    payload = {
        "run_id": "demo",
        "symbol": "BTCUSDT",
        "indicators": {"rsi_dynamic": 61.0, "bb_width_dynamic": 0.04, "trend_strength": 0.3},
        "smc": {"status": "SMC_BULLISH_SETUP"},
        "vision": {"bias": "bullish", "confidence": 0.7},
    }
    print(agent.run(payload).output)


if __name__ == "__main__":
    main()
