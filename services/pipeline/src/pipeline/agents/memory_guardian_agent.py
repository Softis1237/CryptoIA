from __future__ import annotations

"""Memory Guardian Agent — централизованное управление структурированными уроками."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..infra import db
from ..infra.run_lock import slot_lock
from .base import AgentResult, BaseAgent
from .embeddings import embed_text


class StructuredLesson(BaseModel):
    scope: str = "trading"
    error_type: str
    market_regime: str
    involved_agents: List[str] = Field(default_factory=list)
    triggering_signals: List[str] = Field(default_factory=list)
    key_factors_missed: List[str] = Field(default_factory=list)
    correct_action_suggestion: str
    confidence_before: float
    outcome_after: float
    meta: Dict[str, Any] = Field(default_factory=dict)


class MemoryQuery(BaseModel):
    scope: str = "trading"
    context: Dict[str, Any]
    top_k: int = 5


@dataclass(slots=True)
class MemoryGuardianAgent(BaseAgent):
    name: str = "memory-guardian"
    priority: int = 15

    def run(self, payload: dict) -> AgentResult:
        lesson_payload = payload.get("lesson")
        if not isinstance(lesson_payload, dict):
            raise ValueError("MemoryGuardianAgent requires 'lesson' payload")
        record = StructuredLesson(**lesson_payload)
        hash_val = str(record.meta.get("hash") or self._hash_record(record))
        if "hash" not in record.meta:
            record.meta["hash"] = hash_val
        with slot_lock(f"memory_guardian:{record.scope}"):
            db.insert_structured_lesson(
                scope=record.scope,
                hash_val=hash_val,
                lesson=record.model_dump(),
                error_type=record.error_type,
                market_regime=record.market_regime,
                involved_agents=record.involved_agents,
                triggering_signals=record.triggering_signals,
                key_factors_missed=record.key_factors_missed,
                correct_action_suggestion=record.correct_action_suggestion,
                confidence_before=record.confidence_before,
                outcome_after=record.outcome_after,
            )
            embedding = self._embed(record)
            if embedding:
                db.upsert_structured_lesson_vector(hash_val, embedding)
        return AgentResult(name=self.name, ok=True, output={"hash": hash_val})

    def query(self, payload: dict) -> AgentResult:
        params = MemoryQuery(**payload)
        embedding = self._embed_context(params.context)
        if not embedding:
            return AgentResult(name=f"{self.name}-query", ok=True, output={"lessons": []})
        lessons = db.search_structured_lessons(embedding, params.scope, params.top_k)
        return AgentResult(name=f"{self.name}-query", ok=True, output={"lessons": lessons})

    def refresh(self, scope: str = "trading") -> dict:
        recent = db.fetch_recent_structured_lessons(scope=scope, limit=10)
        return {"lessons": recent}

    def lessons_for_red_team(self, scope: str = "trading", limit: int = 10) -> List[dict]:
        return db.fetch_recent_structured_lessons(scope=scope, limit=limit)

    def _hash_record(self, record: StructuredLesson) -> str:
        payload = record.model_dump()
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _embed(self, record: StructuredLesson) -> List[float] | None:
        text = json.dumps(
            {
                "error": record.error_type,
                "regime": record.market_regime,
                "agents": record.involved_agents,
                "signals": record.triggering_signals,
                "factors": record.key_factors_missed,
                "action": record.correct_action_suggestion,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            return embed_text(text)
        except Exception:
            return None

    def _embed_context(self, context: Dict[str, Any]) -> List[float] | None:
        try:
            text = json.dumps(context, ensure_ascii=False, sort_keys=True)
            return embed_text(text)
        except Exception:
            return None


def main() -> None:
    guardian = MemoryGuardianAgent()
    lesson = {
        "lesson": {
            "error_type": "false_breakout",
            "market_regime": "choppy_range",
            "involved_agents": ["SMC_Analyst", "Debate_Arbiter"],
            "triggering_signals": ["SMC_BULLISH_SETUP"],
            "key_factors_missed": ["liquidity_grab"],
            "correct_action_suggestion": "Ждать подтверждения объёмов",
            "confidence_before": 0.7,
            "outcome_after": -0.04,
        }
    }
    print(guardian.run(lesson).output)
    print(guardian.query({"scope": "trading", "context": {"market_regime": "choppy_range"}}).output)


if __name__ == "__main__":
    main()
