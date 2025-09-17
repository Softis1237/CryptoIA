from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.trading.backtest import (
    BacktestConfig,
    BarEvent,
    ExecutionReport,
    ExchangeSimulator,
    OrderRequest,
    OrderSide,
    OrderType,
    run_from_dataframe,
)
from pipeline.trading.backtest.portfolio import build_portfolio


def test_exchange_simulator_partial_fill() -> None:
    simulator = ExchangeSimulator(maker_fee=0.0001, taker_fee=0.0002, slippage=0.0, volume_limit=0.25)
    bar = BarEvent(
        timestamp=pd.Timestamp("2023-01-01 00:00"),
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        volume=2.0,
    )
    simulator._current_bar = bar
    order = OrderRequest(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=4.0,
        order_type=OrderType.LIMIT,
        price=101.0,
    )
    simulator.submit(order, bar.timestamp)
    fills = simulator.process_bar(bar)
    assert fills, "Ожидался частичный fill"
    fill = fills[0]
    assert fill.quantity == pytest.approx(0.5)  # volume_limit=0.25 * volume=2 -> 0.5 BTC
    assert simulator.open_orders, "Ордер должен оставаться частично открытым"


def test_vector_backtester_runs_and_reports() -> None:
    timestamps = pd.date_range("2022-01-01", periods=60, freq="1h")
    prices = [100 + i * 0.5 for i in range(30)] + [115 - i * 0.4 for i in range(30)]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [10.0] * len(prices),
        }
    )
    config = BacktestConfig(symbol="BTC/USDT", starting_cash=10_000.0, slippage=0.0)
    report = run_from_dataframe(
        df,
        config=config,
        strategy_path="pipeline.trading.backtest.strategies:MovingAverageCrossStrategy",
        strategy_params={"fast": 3, "slow": 8, "risk_fraction": 0.3},
    )
    assert "win_rate" in report.metrics
    assert "sharpe_ratio" in report.metrics
    assert report.equity_curve.shape[0] == len(df)
    assert report.trades, "Стратегия должна совершить хотя бы одну сделку"


def test_portfolio_mark_to_market_uses_latest_price() -> None:
    portfolio = build_portfolio(1_000.0)
    timestamp = pd.Timestamp("2023-01-01 00:00")
    fill = ExecutionReport(
        order_id="1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        price=100.0,
        quantity=1.0,
        fee=0.0,
        timestamp=timestamp,
        is_partial=False,
        remaining=0.0,
    )
    portfolio.process_fill(fill)
    expected_cash = portfolio.cash

    initial_snapshot = portfolio.mark_to_market(timestamp, store=False)
    assert initial_snapshot.equity == pytest.approx(1_000.0)
    assert initial_snapshot.positions_value == pytest.approx(100.0)

    portfolio.update_mark_price("BTC/USDT", 110.0)
    next_snapshot = portfolio.mark_to_market(timestamp + pd.Timedelta(hours=1), store=False)
    assert next_snapshot.equity == pytest.approx(expected_cash + 110.0)
    assert next_snapshot.positions_value == pytest.approx(110.0)
    exposure = next_snapshot.metadata["exposure"]["BTC/USDT"]
    assert exposure["last_price"] == pytest.approx(110.0)
    assert exposure["market_value"] == pytest.approx(110.0)
