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
