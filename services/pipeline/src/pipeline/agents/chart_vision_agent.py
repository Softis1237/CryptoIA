from __future__ import annotations

"""Chart Vision Agent — мультитаймфреймовый визуальный анализ."""

import os
from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field

from ..infra import metrics
from ..infra.run_lock import slot_lock
from ..reasoning.llm import call_openai_json
from .base import AgentResult, BaseAgent


class ChartVisionPayload(BaseModel):
    run_id: str
    symbol: str
    slot: str = "chart_vision"
    image_urls: List[str] = Field(..., min_items=1)
    regime: str = "range"


@dataclass(slots=True)
class VisionInsight:
    timeframe: str
    bias: str
    confidence: float
    notes: List[str]


class ChartVisionAgent(BaseAgent):
    name = "chart-vision-agent"
    priority = 32

    def run(self, payload: dict) -> AgentResult:
        params = ChartVisionPayload(**payload)
        with slot_lock(params.slot):
            insights = self._analyze(params)
            llm_insights = self._query_llm(params)
            if llm_insights:
                insights = llm_insights
            merged = self._merge(insights)
            metrics.push_values(
                job="chart_vision_agent",
                values={"chart_vision_bias": 1.0 if merged["bias"] == "bullish" else -1.0},
                labels={"symbol": params.symbol},
            )
        return AgentResult(name=self.name, ok=True, output=merged)

    def _analyze(self, params: ChartVisionPayload) -> List[VisionInsight]:
        # Deterministic эвристика на основе порядка URL
        insights: List[VisionInsight] = []
        bias_cycle = ["bullish", "bearish", "neutral"]
        for idx, url in enumerate(params.image_urls):
            tf = self._infer_timeframe(url, idx)
            bias = bias_cycle[idx % len(bias_cycle)]
            if params.regime.startswith("trend_up"):
                bias = "bullish"
            elif params.regime.startswith("trend_down"):
                bias = "bearish"
            confidence = max(0.2, min(0.9, 0.5 + 0.1 * (len(params.image_urls) - idx)))
            notes = [f"url:{url}", f"regime:{params.regime}"]
            insights.append(VisionInsight(timeframe=tf, bias=bias, confidence=confidence, notes=notes))
        return insights

    def _merge(self, insights: List[VisionInsight]) -> dict:
        weights = {"1w": 1.2, "1d": 1.0, "4h": 0.9, "1h": 0.8}
        bulls = 0.0
        bears = 0.0
        total_conf = 0.0
        for i in insights:
            w = weights.get(i.timeframe.lower(), 0.7)
            if i.bias == "bullish":
                bulls += i.confidence * w
            elif i.bias == "bearish":
                bears += i.confidence * w
            total_conf += i.confidence * w
        if bulls > bears * 1.1:
            bias = "bullish"
        elif bears > bulls * 1.1:
            bias = "bearish"
        else:
            bias = "neutral"
        avg_conf = total_conf / max(1, len(insights))
        return {
            "bias": bias,
            "confidence": round(avg_conf, 3),
            "insights": [i.__dict__ for i in insights],
        }

    def _query_llm(self, params: ChartVisionPayload) -> List[VisionInsight] | None:
        flag = os.getenv("ENABLE_CHART_VISION_LLM", "0")
        if flag not in {"1", "true", "True"}:
            return None
        try:
            sys_prompt = (
                "You are a multimodal technical analyst."
                " Analyse the supplied chart image URLs on multiple timeframes."
                " Return JSON with key 'insights': list of objects"
                " {timeframe, bias (bullish/bearish/neutral), confidence (0..1), notes[]}"
                " Bias should reflect the dominant signal across frames, confidence must be calibrated to 0..1."
            )
            user_content = [
                {
                    "type": "text",
                    "text": (
                        "Evaluate the technical context for symbol {symbol} in regime {regime}. "
                        "Identify confluence or conflicts across provided timeframes."
                    ).format(symbol=params.symbol, regime=params.regime),
                }
            ]
            for url in params.image_urls:
                user_content.append({"type": "image_url", "image_url": {"url": url}})
            raw = call_openai_json(
                sys_prompt,
                user_content,
                model=os.getenv("OPENAI_MODEL_VISION") or os.getenv("OPENAI_MODEL_MASTER"),
            )
            items = raw.get("insights") if isinstance(raw, dict) else None
            result: List[VisionInsight] = []
            for item in items or []:
                try:
                    result.append(
                        VisionInsight(
                            timeframe=str(item.get("timeframe") or "hybrid"),
                            bias=str(item.get("bias") or "neutral").lower(),
                            confidence=float(item.get("confidence") or 0.5),
                            notes=[str(n) for n in (item.get("notes") or [])],
                        )
                    )
                except Exception:
                    continue
            return result or None
        except Exception:
            return None

    def _infer_timeframe(self, url: str, idx: int) -> str:
        for tf in ("1m", "5m", "15m", "1h", "4h", "1d", "1w"):
            if tf in url.lower():
                return tf
        order = ["1w", "1d", "4h", "1h", "15m", "5m", "1m"]
        return order[idx % len(order)]


def main() -> None:
    agent = ChartVisionAgent()
    sample = {
        "run_id": "demo",
        "symbol": "BTCUSDT",
        "image_urls": ["s3://charts/BTCUSDT_4h.png", "s3://charts/BTCUSDT_1h.png"],
    }
    print(agent.run(sample).output)


if __name__ == "__main__":
    main()
