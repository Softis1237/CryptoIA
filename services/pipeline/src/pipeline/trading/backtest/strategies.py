from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Optional, Type

import pandas as pd

from .types import BarEvent, OrderRequest, OrderSide, OrderType, PortfolioState, Strategy


class MovingAverageCrossStrategy(Strategy):
    """Простая лонг-стратегия пересечения скользящих средних."""

    def __init__(self, fast: int = 12, slow: int = 26, risk_fraction: float = 0.2) -> None:
        if fast >= slow:
            raise ValueError("fast период должен быть меньше slow")
        self.fast = fast
        self.slow = slow
        self.risk_fraction = risk_fraction
        self._history: list[float] = []
        self._position: str = "flat"
        self._current_qty: float = 0.0

    def on_bar(
        self,
        bar: BarEvent,
        history: Iterable[BarEvent],
        portfolio_state: PortfolioState,
    ) -> list[OrderRequest]:
        self._history.append(bar.close)
        if len(self._history) < self.slow:
            return []
        prices = pd.Series(self._history[-self.slow :])
        fast_ma = prices.ewm(span=self.fast, adjust=False).mean().iloc[-1]
        slow_ma = prices.ewm(span=self.slow, adjust=False).mean().iloc[-1]
        orders: list[OrderRequest] = []
        symbol = portfolio_state.metadata.get("config_symbol", bar.extras.get("symbol", "BTC/USDT"))
        if fast_ma > slow_ma and self._position != "long":
            qty = self._position_size(portfolio_state.equity, bar.close)
            if qty > 0:
                orders.append(
                    OrderRequest(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=qty,
                        order_type=OrderType.MARKET,
                    )
                )
                self._position = "long"
                self._current_qty = qty
        elif fast_ma < slow_ma and self._position == "long":
            qty = round(self._current_qty, 6)
            if qty > 0:
                orders.append(
                    OrderRequest(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=qty,
                        order_type=OrderType.MARKET,
                    )
                )
                self._position = "flat"
                self._current_qty = 0.0
        return orders

    def _position_size(self, equity: float, price: float) -> float:
        target_value = equity * self.risk_fraction
        if target_value <= 0 or price <= 0:
            return 0.0
        qty = target_value / price
        return round(qty, 6)

    def finalize(self) -> None:  # pragma: no cover - простая стратегия
        return None


def load_strategy(path: str, params: Optional[Dict[str, Any]] = None) -> Strategy:
    """Импортирует стратегию по пути вида module:ClassName."""

    params = params or {}
    if ":" not in path:
        raise ValueError("Ожидался путь вида 'module:ClassName'")
    module_name, class_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    strategy_cls: Type[Strategy] = getattr(module, class_name)
    return strategy_cls(**params)
