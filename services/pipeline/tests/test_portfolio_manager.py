from __future__ import annotations

import types

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.trading.portfolio_manager import PortfolioManager, DBStorage, Position


class _FakeCursor:
    def __init__(self) -> None:
        self._last_sql = ""
        self._last_args = ()
        self._phase = None

    def execute(self, sql: str, args: tuple) -> None:  # noqa: D401
        self._last_sql = sql
        self._last_args = args
        if "FROM portfolio_positions" in sql:
            self._phase = "get_position"
        elif "portfolio_equity_daily" in sql:
            self._phase = "equity"
        elif "portfolio_risk_exposure" in sql:
            self._phase = "risk"
        else:
            self._phase = "upsert"

    def fetchone(self):  # noqa: D401
        if self._phase == "get_position":
            # symbol, direction, quantity, avg_price, unrealized_pnl
            return ("BTC/USDT", "long", 0.25, 50000.0, 250.0)
        if self._phase == "equity":
            return (10000.0,)
        if self._phase == "risk":
            return (123.0,)
        return None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc):  # noqa: D401
        return False


class _FakeConn:
    def cursor(self) -> _FakeCursor:  # noqa: D401
        return _FakeCursor()

    def __enter__(self) -> "_FakeConn":
        return self

    def __exit__(self, *exc):  # noqa: D401
        return False


def _fake_get_conn():  # noqa: D401
    return _FakeConn()


def test_portfolio_manager_db_branch(monkeypatch):
    db = DBStorage()
    # monkeypatch the internal connection getter
    monkeypatch.setattr(db, "_get_conn", _fake_get_conn)
    pm = PortfolioManager(storage=db)

    pos = pm.get_position("BTC/USDT")
    assert isinstance(pos, Position)
    assert pos.direction == "long"
    assert pos.quantity == 0.25

    # sizing by fixed risk
    qty = pm.position_size_by_risk(10_000.0, 0.01, 50_000.0, 49_500.0)
    assert qty > 0

    # decision policy
    decision, reason = pm.decide_with_existing("BTC/USDT", "LONG", 0.7)
    assert decision in {"scale_in", "open", "ignore"}
    # opposite side must be ignored
    decision2, reason2 = pm.decide_with_existing("BTC/USDT", "SHORT", 0.7)
    assert decision2 == "ignore"

    # equity/risk
    assert pm.total_equity() == 10000.0
    assert pm.total_risk() == 123.0
