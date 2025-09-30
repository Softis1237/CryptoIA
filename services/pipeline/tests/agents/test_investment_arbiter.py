from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ARB_LOG_TO_DB", "0")
os.environ.setdefault("ARB_STORE_S3", "0")

from pipeline.agents.investment_arbiter import InvestmentArbiter
from pipeline.agents.base import AgentResult


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


def test_arbiter_decide_with_context():
    class StubContext:
        def run(self, payload):
            return AgentResult(
                name="context", ok=True, output={"context": {"text": "macro", "summary": {"body": "ok"}, "tokens_estimate": 256}}
            )

    class StubCritique:
        def run(self, payload):
            return AgentResult(
                name="critique",
                ok=True,
                output={
                    "counterarguments": ["объём падает"],
                    "invalidators": ["пробой 60k"],
                    "missing_factors": [],
                    "probability_adjustment": -5,
                    "recommendation": "REVISE",
                },
            )

    def fake_llm(system_prompt, user_prompt, model=None, temperature=0.2):
        return {
            "scenario": "LONG",
            "probability_pct": 68,
            "macro_summary": "ETF приток",
            "technical_summary": "Цена у поддержки",
            "contradictions": ["новости бычьи, но объём падает"],
            "explanation": "Приток средств компенсирует продажи",
        }

    arbiter = InvestmentArbiter(
        trust_weight=0.0,
        lesson_weight=0.0,
        context_agent=StubContext(),
        critique_agent=StubCritique(),
        analyst_llm=fake_llm,
    )

    out = arbiter.decide(
        {
            "run_id": "test",
            "slot": "manual",
            "symbol": "BTC/USDT",
            "planned_side": "LONG",
            "regime_label": "trend_up",
            "lessons": [],
            "model_trust": {},
            "risk_flags": [],
            "safe_mode": False,
            "base_proba_up": 0.6,
            "context_payload": {},
        }
    )
    assert out.analysis is not None
    assert out.analysis.scenario == "LONG"
    assert out.critique is not None
    assert out.evaluation.risk_stance in {"risk_on", "risk_watch"}
    assert any(note.startswith("analysis_scenario") for note in out.evaluation.notes)


def test_arbiter_accounts_for_trust_weights():
    arbiter = InvestmentArbiter(acceptance_threshold=0.6, trust_weight=0.15)
    decision_high = arbiter.evaluate(
        base_proba_up=0.62,
        side="LONG",
        regime="range",
        lessons=[],
        model_trust={"ml:ensemble": 0.9, "ets": 0.85},
        risk_flags=[],
        safe_mode=False,
    )
    decision_low = arbiter.evaluate(
        base_proba_up=0.62,
        side="LONG",
        regime="range",
        lessons=[],
        model_trust={"ml:ensemble": 0.2, "ets": 0.25},
        risk_flags=[],
        safe_mode=False,
    )
    assert decision_high.success_probability > decision_low.success_probability
    assert any(note.startswith("trust_") for note in decision_high.notes)


def test_arbiter_warning_reduces_probability(monkeypatch):
    class StubContext:
        def run(self, payload):
            return AgentResult(
                name="context",
                ok=True,
                output={"context": {"text": "macro", "summary": {"body": "ok"}, "tokens_estimate": 256}},
            )

    class StubCritique:
        def run(self, payload):
            return AgentResult(
                name="critique",
                ok=True,
                output={
                    "counterarguments": [""],
                    "invalidators": [],
                    "missing_factors": [],
                    "probability_adjustment": 0,
                    "recommendation": "CONFIRM",
                },
            )

    def fake_llm(system_prompt, user_prompt, model=None, temperature=0.2):
        return {
            "scenario": "LONG",
            "probability_pct": 70,
            "macro_summary": "short",
            "technical_summary": "short",
            "contradictions": ["strong resistance"],
            "explanation": "",
        }

    arbiter = InvestmentArbiter(
        context_agent=StubContext(),
        critique_agent=StubCritique(),
        analyst_llm=fake_llm,
    )

    out = arbiter.decide(
        {
            "run_id": "warn",
            "slot": "manual",
            "symbol": "BTC/USDT",
            "planned_side": "LONG",
            "regime_label": "trend_up",
            "lessons": [],
            "model_trust": {},
            "risk_flags": [],
            "safe_mode": False,
            "base_proba_up": 0.6,
            "context_payload": {},
        }
    )
    # warnings should cause probability drop relative to baseline 0.6
    assert out.evaluation.success_probability < 0.6
    assert any(note.startswith("analysis_warn") for note in out.evaluation.notes)
