import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pipeline.rt.alert_priority import mark_and_score
from pipeline.trading.confidence import compute


def test_alert_priority_scoring_simple(monkeypatch):
    # No redis available in test env — just ensure function returns a label
    pr = mark_and_score("MOMENTUM", window_sec=1)
    assert pr.label in {"low", "medium", "high", "critical"}


def test_confidence_aggregation():
    base = 0.5
    ctx = {
        "trigger_type": "MOMENTUM",
        "ta_sentiment": "bullish",
        "regime_label": "trend_up",
        "side": "LONG",
        "pattern_hit": True,
        "deriv_signal": False,
        "news_impact": 0.2,
        "alert_priority": "high",
        "smc_status": "SMC_BULLISH_SETUP",
        "whale_status": "WHALE_BULLISH",
    }
    out = compute(base, ctx)
    assert 0.5 <= out.value <= 1.0
    assert isinstance(out.factors, dict)

