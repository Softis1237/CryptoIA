from __future__ import annotations

import os

import pytest

from pipeline.agents.context_builder_agent import ContextBuilderAgent


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch):
    monkeypatch.setenv("CONTEXT_FETCH_SMC", "0")
    monkeypatch.setenv("CONTEXT_FETCH_TA", "0")
    monkeypatch.setenv("CONTEXT_CACHE_TTL", "60")
    yield


def test_context_builder_composes_sections(monkeypatch):
    monkeypatch.setattr(
        "pipeline.agents.context_builder_agent.get_relevant_lessons",
        lambda ctx, k=5: [{"title": "Test lesson", "action": "Avoid chasing"}],
    )
    monkeypatch.setattr(
        "pipeline.agents.context_builder_agent.fetch_latest_strategic_verdict",
        lambda *args, **kwargs: {},
    )
    agent = ContextBuilderAgent()
    res = agent.run(
        {
            "run_id": "demo",
            "symbol": "BTC/USDT",
            "regime": {"label": "trend_up", "confidence": 0.6},
            "news": [{"title": "ETF approved", "sentiment": "positive", "impact_score": 0.9}],
            "onchain": [{"metric": "netflow", "value": -1200}],
            "lessons_context": {"regime": "trend_up"},
            "smc": {"status": "SMC_BULLISH_SETUP", "trend_4h": "trend_up", "zones": {"1h": []}},
            "advanced_ta": {"swing_high": 120, "swing_low": 100, "bias": "up_swing", "fib_support": [110]},
        }
    )
    context = res.output["context"]
    assert any(section["title"] == "Новости и макро" for section in context["sections"])
    assert any("LLM недоступен" not in section["body"] for section in context["sections"])
    assert context["tokens_estimate"] > 0
