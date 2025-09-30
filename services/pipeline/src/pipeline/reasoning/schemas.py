from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ExplainResponse(BaseModel):
    bullets: List[str] = Field(default_factory=list)


class DebateResponse(BaseModel):
    bullets: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class ScenarioItem(BaseModel):
    if_level: str
    then_path: str
    prob: float
    invalidation: str


class ScenarioResponse(BaseModel):
    scenarios: List[ScenarioItem]
    levels: Optional[List[float]] = None


class ArbiterAnalystResponse(BaseModel):
    scenario: str = Field(..., pattern="^(LONG|SHORT|FLAT)$", description="Final bias")
    probability_pct: float = Field(..., ge=0, le=100)
    macro_summary: str = Field(..., max_length=1500)
    technical_summary: str = Field(..., max_length=1500)
    contradictions: List[str] = Field(default_factory=list)
    explanation: str = Field(..., max_length=2000)

    def sanitized(self) -> "ArbiterAnalystResponse":
        data = self.model_dump()
        data["contradictions"] = [c[:300] for c in data["contradictions"]][:6]
        return ArbiterAnalystResponse(**data)


class CritiqueResponse(BaseModel):
    counterarguments: List[str] = Field(default_factory=list)
    invalidators: List[str] = Field(default_factory=list)
    missing_factors: List[str] = Field(default_factory=list)
    probability_adjustment: float = Field(default=0.0, ge=-50, le=50)
    recommendation: str = Field(..., pattern="^(CONFIRM|REVISE|REJECT)$")

    def sanitized(self) -> "CritiqueResponse":
        data = self.model_dump()
        data["counterarguments"] = [c[:300] for c in data["counterarguments"]][:6]
        data["invalidators"] = [c[:200] for c in data["invalidators"]][:6]
        data["missing_factors"] = [c[:200] for c in data["missing_factors"]][:6]
        return CritiqueResponse(**data)
