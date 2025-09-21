from __future__ import annotations

import json
import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.orchestration.agent_flow import TradeAgent
from pipeline.orchestration.resource_policy import ResourcePolicy


def test_resource_policy_switches_to_safe_mode_from_pending_events():
    policy = ResourcePolicy()
    plan = policy.build_plan(
        events=[],
        slot="test",
        pending_events=[{"event_type": "data_anomaly"}],
    )
    assert plan.mode == "High-Alert"
    assert plan.safe_mode is True
    assert plan.risk_overrides["risk_per_trade"] <= 0.003
    assert plan.risk_overrides["leverage_cap"] <= 5


def test_trade_agent_respects_safe_mode_overrides(monkeypatch):
    agent = TradeAgent()
    payload = {
        "current_price": 30000.0,
        "y_hat_4h": 100.0,
        "interval_low": -50.0,
        "interval_high": 250.0,
        "proba_up": 0.6,
        "atr": 500.0,
        "regime": "trend_up",
        "account_equity": 10000.0,
        "risk_per_trade": 0.01,
        "leverage_cap": 20.0,
        "now_ts": 0,
        "tz_name": "UTC",
        "valid_for_minutes": 90,
        "horizon_minutes": 240,
    }

    class DummyCard(dict):
        pass

    def fake_run_trade(input_obj):
        return DummyCard({"position": "LONG", "risk_per_trade": input_obj.risk_per_trade})

    def fake_verify(card, **kwargs):
        return True, "ok"

    monkeypatch.setattr("pipeline.orchestration.agent_flow.run_trade", fake_run_trade)
    monkeypatch.setattr("pipeline.orchestration.agent_flow.verify", fake_verify)

    os.environ["SAFE_MODE"] = "1"
    os.environ["SAFE_MODE_RISK_OVERRIDES"] = json.dumps({"risk_per_trade": 0.002, "leverage_cap": 4})
    try:
        result = agent.run(payload)
        assert result.ok is True
        out = result.output
        assert out["card"]["risk_per_trade"] <= 0.002
        assert out["verified"] is True
    finally:
        os.environ.pop("SAFE_MODE", None)
        os.environ.pop("SAFE_MODE_RISK_OVERRIDES", None)
