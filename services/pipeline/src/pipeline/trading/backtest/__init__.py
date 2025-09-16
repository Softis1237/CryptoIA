"""Vector backtesting utilities for CryptoIA trading pipeline."""

from .types import (
    BacktestConfig,
    BarEvent,
    ExecutionReport,
    OrderRequest,
    OrderSide,
    OrderType,
    TradeRecord,
)
from .portfolio import PortfolioState
from .exchange_simulator import ExchangeSimulator
from .runner import VectorBacktester, run_from_dataframe
from .report import BacktestReport

__all__ = [
    "BacktestConfig",
    "BacktestReport",
    "BarEvent",
    "ExchangeSimulator",
    "ExecutionReport",
    "OrderRequest",
    "OrderSide",
    "OrderType",
    "PortfolioState",
    "TradeRecord",
    "VectorBacktester",
    "run_from_dataframe",
]
