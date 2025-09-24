from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.trading.risk_loop_live import GuardianCandidate, PositionGuardianAgent


def test_position_guardian_alerts_when_success_drops(monkeypatch):
    agent = PositionGuardianAgent(symbol="BTC/USDT", prob_threshold=0.4, price_fetcher=lambda _: 0.0)

    candidate = GuardianCandidate(
        symbol="BTC/USDT",
        side="LONG",
        entry=100.0,
        sl=90.0,
        tp=130.0,
        qty=1.0,
        opened_at=datetime.now(timezone.utc) - timedelta(minutes=120),
        current_price=86.0,
        source="paper",
        run_id="run123",
        risk_profile={"hold_minutes": 45},
        base_probability=0.5,
    )

    alerts: list[dict] = []
    history = {
        "LONG": {
            "success_minutes": [30.0, 28.0],
            "failure_minutes": [60.0],
            "success_rate": 0.66,
            "failure_rate": 0.34,
        }
    }
    monkeypatch.setattr(agent, "_collect_candidates", lambda: [candidate])
    monkeypatch.setattr(agent, "_notify", lambda assessment: alerts.append(assessment))
    monkeypatch.setattr(agent, "_log_event", lambda assessment: alerts.append({"logged": assessment["run_id"]}))
    monkeypatch.setattr(agent, "_fetch_history_stats", lambda: history)

    result = agent.run()

    assert result["checked"] == 1
    assert result["alerts"], "guardian should raise alert"
    assert alerts  # notification triggered
    assert result["alerts"][0]["should_alert"] is True
