from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.agents.investment_arbiter import InvestmentArbiter


def test_arbiter_penalizes_negative_lessons():
    arbiter = InvestmentArbiter(acceptance_threshold=0.7, lesson_weight=0.1)
    lessons = [
        {
            "lesson": {
                "market_regime": "trend_up",
                "involved_agents": ["ML Ensemble"],
                "outcome_after": -0.05,
            }
        }
    ]
    decision = arbiter.evaluate(
        base_proba_up=0.8,
        side="LONG",
        regime="trend_up",
        lessons=lessons,
        risk_flags=[],
        safe_mode=False,
    )
    assert decision.success_probability < 0.8
    assert "lesson_penalty:models" in decision.notes


def test_arbiter_flags_risk_on_above_threshold():
    arbiter = InvestmentArbiter(acceptance_threshold=0.7)
    decision = arbiter.evaluate(
        base_proba_up=0.75,
        side="LONG",
        regime="trend_up",
        lessons=[],
        risk_flags=[],
        safe_mode=False,
    )
    assert decision.risk_stance == "risk_on"
    assert decision.confidence_floor is not None
