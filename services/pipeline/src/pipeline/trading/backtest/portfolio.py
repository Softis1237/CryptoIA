from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .types import ExecutionReport, OrderSide, PortfolioSnapshot, PortfolioState, TradeRecord


@dataclass
class _Lot:
    timestamp: pd.Timestamp
    price: float
    quantity: float


@dataclass
class _Position:
    symbol: str
    direction: Optional[str] = None  # "long" | "short"
    lots: List[_Lot] = field(default_factory=list)

    def total_quantity(self) -> float:
        return sum(l.quantity for l in self.lots)

    def average_price(self) -> float:
        total_qty = self.total_quantity()
        if total_qty == 0:
            return 0.0
        weighted = sum(l.price * l.quantity for l in self.lots)
        return weighted / total_qty


class Portfolio:
    """Учет денежных средств и открытых позиций."""

    def __init__(self, starting_cash: float) -> None:
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: Dict[str, _Position] = {}
        self.realized_pnl = 0.0
        self.history: List[PortfolioSnapshot] = []
        self.trades: List[TradeRecord] = []
        self._last_prices: Dict[str, float] = {}

    def update_mark_price(self, symbol: str, price: float) -> None:
        """Сохраняет последнюю рыночную цену инструмента."""

        if not symbol:
            return
        self._last_prices[symbol] = price

    # ------------------------------------------------------------------
    def process_fill(self, fill: ExecutionReport) -> None:
        position = self.positions.setdefault(fill.symbol, _Position(symbol=fill.symbol))
        timestamp = fill.timestamp
        price = fill.price
        quantity = fill.quantity
        side = fill.side
        fee = fill.fee

        if position.direction is None:
            # Открытие новой позиции
            position.direction = "long" if side == OrderSide.BUY else "short"

        is_same_direction = (
            (position.direction == "long" and side == OrderSide.BUY)
            or (position.direction == "short" and side == OrderSide.SELL)
        )

        self._apply_cash_flow(price, quantity, side, fee)

        if is_same_direction:
            self._open_lot(position, timestamp, price, quantity)
        else:
            self._close_lot(position, timestamp, price, quantity, side, fee)

        if position.total_quantity() == 0:
            position.direction = None

        self._last_prices[fill.symbol] = price

    def mark_to_market(self, timestamp: pd.Timestamp, store: bool = True) -> PortfolioSnapshot:
        positions_value = 0.0
        exposure: Dict[str, Dict[str, float]] = {}
        for symbol, pos in self.positions.items():
            qty = pos.total_quantity()
            if qty == 0:
                continue
            last = self._last_prices.get(symbol, pos.average_price())
            if pos.direction == "long":
                pnl = (last - pos.average_price()) * qty
                market_value = last * qty
            else:
                pnl = (pos.average_price() - last) * qty
                market_value = -last * qty
            positions_value += market_value
            exposure[symbol] = {
                "direction": pos.direction or "flat",
                "quantity": qty,
                "avg_price": pos.average_price(),
                "last_price": last,
                "unrealized_pnl": pnl,
                "market_value": market_value,
                "symbol": symbol,
            }
        equity = self.cash + positions_value
        leverage = 0.0
        notional = sum(abs(ex["last_price"] * ex["quantity"]) for ex in exposure.values())
        if equity > 0:
            leverage = notional / equity
        snap = PortfolioSnapshot(
            timestamp=timestamp,
            equity=equity,
            cash=self.cash,
            positions_value=positions_value,
            leverage=leverage,
            metadata={"exposure": exposure},
        )
        if store:
            self.history.append(snap)
        return snap

    # ------------------------------------------------------------------
    def _open_lot(self, position: _Position, timestamp: pd.Timestamp, price: float, quantity: float) -> None:
        position.lots.append(_Lot(timestamp=timestamp, price=price, quantity=quantity))

    def _close_lot(
        self,
        position: _Position,
        timestamp: pd.Timestamp,
        price: float,
        quantity: float,
        side: OrderSide,
        fee: float,
    ) -> None:
        remaining = quantity
        pnl_total = 0.0
        while remaining > 1e-12 and position.lots:
            lot = position.lots[0]
            closable = min(lot.quantity, remaining)
            pnl = self._calc_pnl(position.direction, lot.price, price, closable)
            pnl_total += pnl
            lot.quantity -= closable
            remaining -= closable
            if lot.quantity <= 1e-12:
                position.lots.pop(0)
            exposure_side = "LONG" if position.direction == "long" else "SHORT"
            trade = TradeRecord(
                symbol=position.symbol,
                side=exposure_side,
                entry_time=lot.timestamp,
                exit_time=timestamp,
                entry_price=lot.price,
                exit_price=price,
                quantity=closable,
                pnl=pnl - fee * (closable / quantity),
                return_pct=(price - lot.price) / lot.price if exposure_side == "LONG" else (lot.price - price) / lot.price,
            )
            self.trades.append(trade)
        self.realized_pnl += pnl_total - fee
        if remaining > 1e-12:
            # разворот
            new_direction = "long" if side == OrderSide.BUY else "short"
            position.direction = new_direction
            position.lots.append(_Lot(timestamp=timestamp, price=price, quantity=remaining))

    def _apply_cash_flow(self, price: float, quantity: float, side: OrderSide, fee: float) -> None:
        notional = price * quantity
        if side == OrderSide.BUY:
            self.cash -= notional + fee
        else:
            self.cash += notional - fee

    def _calc_pnl(self, direction: Optional[str], entry: float, exit: float, quantity: float) -> float:
        if direction == "long":
            return (exit - entry) * quantity
        elif direction == "short":
            return (entry - exit) * quantity
        return 0.0

    # ------------------------------------------------------------------
    def snapshot_state(self) -> PortfolioState:
        if not self.history:
            raise RuntimeError("Портфель ещё не сформировал снимков")
        return PortfolioState.from_snapshot(self.history[-1])

    def get_trades(self) -> List[TradeRecord]:
        return self.trades


def build_portfolio(starting_cash: float) -> Portfolio:
    return Portfolio(starting_cash=starting_cash)
