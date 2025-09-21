from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.agents.red_team_agent import RedTeamAgent
from pipeline.trading.backtest.report import BacktestReport
from pipeline.trading.backtest.types import BacktestConfig, TradeRecord


class DummyGuardian:
    def lessons_for_red_team(self, scope: str = "trading", limit: int = 5):
        return [
            {
                "lesson": {
                    "error_type": "false_breakout",
                    "market_regime": "choppy_range",
                    "planned_bias": "bearish",
                    "symbol": "BTCUSDT",
                    "correct_action_suggestion": "Ожидать подтверждения объёмов",
                    "confidence_before": 0.65,
                    "outcome_after": -0.04,
                }
            }
        ]


class DummyReport(BacktestReport):
    pass


def test_red_team_agent_generates_report(monkeypatch):
    def fake_run_from_dataframe(df, config: BacktestConfig, strategy_path: str, strategy_params=None):
        curve = pd.DataFrame({"timestamp": df["timestamp"], "equity": 10000 + df.index})
        trades = [
            TradeRecord(
                symbol=config.symbol,
                side="LONG",
                entry_time=df["timestamp"].iloc[0],
                exit_time=df["timestamp"].iloc[-1],
                entry_price=float(df["open"].iloc[0]),
                exit_price=float(df["close"].iloc[-1]),
                quantity=1.0,
                pnl=-120.0,
                return_pct=-0.012,
            )
        ]
        return BacktestReport(
            equity_curve=curve,
            trades=trades,
            metrics={"total_return": -0.03, "win_rate": 0.4},
        )

    monkeypatch.setattr("pipeline.agents.red_team_agent.run_from_dataframe", fake_run_from_dataframe)
    agent = RedTeamAgent(guardian=DummyGuardian())
    output = agent.run({"run_id": "unit-test"}).output
    assert output["suggestions"], "Ожидались рекомендации Красной Команды"
    scenario = output["scenarios"][0]
    assert scenario["metrics"]["total_return"] == -0.03
    assert scenario["parameters"]["stress_factor"] >= 1.0
