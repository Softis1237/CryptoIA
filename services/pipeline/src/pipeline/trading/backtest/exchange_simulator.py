from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional

import pandas as pd

from .types import BarEvent, ExecutionReport, OrderRequest, OrderSide, OrderType


@dataclass
class _SimulatedOrder:
    """Ордер, находящийся в книге симулятора."""

    id: str
    request: OrderRequest
    remaining: float
    created_at: pd.Timestamp
    last_update: pd.Timestamp
    status: str = "open"
    fills: List[ExecutionReport] = field(default_factory=list)


import random


class ExchangeSimulator:
    """Простая симуляция биржи с учётом комиссий и проскальзывания."""

    def __init__(
        self,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        slippage: float = 0.0005,
        volume_limit: float = 0.25,
    ) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.volume_limit = volume_limit
        self._open_orders: Dict[str, _SimulatedOrder] = {}
        self._current_bar: Optional[BarEvent] = None
        # jitter in basis points (0 disables randomness)
        try:
            self._slippage_jitter_bps = float(os.getenv("BT_SLIPPAGE_JITTER_BPS", "0"))  # type: ignore[name-defined]
        except Exception:
            self._slippage_jitter_bps = 0.0

    def reset(self) -> None:
        self._open_orders.clear()
        self._current_bar = None

    def submit(self, request: OrderRequest, current_time: pd.Timestamp) -> _SimulatedOrder:
        """Добавляет ордер в книгу. Маркет-ордера исполняются мгновенно."""

        order_id = request.metadata.get("order_id") if request.metadata else None
        order_id = order_id or uuid.uuid4().hex
        sim = _SimulatedOrder(
            id=order_id,
            request=request,
            remaining=float(request.quantity),
            created_at=current_time,
            last_update=current_time,
        )
        if request.order_type == OrderType.MARKET:
            fills = self._execute_market(sim, current_time)
            sim.fills.extend(fills)
            sim.status = "filled" if sim.remaining == 0.0 else "partially_filled"
        else:
            self._open_orders[sim.id] = sim
        return sim

    def cancel(self, order_id: str) -> None:
        self._open_orders.pop(order_id, None)

    def process_bar(self, bar: BarEvent) -> List[ExecutionReport]:
        """Обновляет книгу и возвращает список исполнений за бар."""

        self._current_bar = bar
        fills: List[ExecutionReport] = []
        for order in list(self._open_orders.values()):
            events = self._try_fill(order, bar)
            fills.extend(events)
            if math.isclose(order.remaining, 0.0, abs_tol=1e-9):
                order.status = "filled"
                self._open_orders.pop(order.id, None)
            elif events:
                order.status = "partially_filled"
        return fills

    # --- internals -----------------------------------------------------
    def _execute_market(self, order: _SimulatedOrder, ts: pd.Timestamp) -> List[ExecutionReport]:
        bar = self._current_bar
        if bar is None:
            raise RuntimeError("Market order cannot be executed without текущего бара")
        available = max(bar.volume * self.volume_limit, 1e-12)
        qty = min(order.remaining, available)
        price = self._apply_slippage(bar.close, order.request.side)
        fee = price * qty * self.taker_fee
        order.remaining = max(order.remaining - qty, 0.0)
        report = ExecutionReport(
            order_id=order.id,
            symbol=order.request.symbol,
            side=order.request.side,
            price=price,
            quantity=qty,
            fee=fee,
            timestamp=ts,
            is_partial=order.remaining > 0,
            remaining=order.remaining,
            metadata={"type": "market"},
        )
        order.fills.append(report)
        return [report]

    def _try_fill(self, order: _SimulatedOrder, bar: BarEvent) -> List[ExecutionReport]:
        if order.remaining <= 0:
            return []
        req = order.request
        if req.order_type == OrderType.LIMIT:
            should_fill = self._limit_hit(req, bar)
            is_maker = True
            trigger_price = req.price
        elif req.order_type == OrderType.STOP_MARKET:
            should_fill = self._stop_triggered(req, bar)
            is_maker = False
            trigger_price = req.stop_price
        else:
            return []
        if not should_fill:
            return []
        available = max(bar.volume * self.volume_limit, 1e-12)
        qty = min(order.remaining, available)
        price = self._fill_price(req, bar, is_maker)
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = price * qty * fee_rate
        order.remaining = max(order.remaining - qty, 0.0)
        order.last_update = bar.timestamp
        report = ExecutionReport(
            order_id=order.id,
            symbol=req.symbol,
            side=req.side,
            price=price,
            quantity=qty,
            fee=fee,
            timestamp=bar.timestamp,
            is_partial=order.remaining > 0,
            remaining=order.remaining,
            metadata={
                "type": req.order_type.value,
                "trigger_price": trigger_price,
            },
        )
        order.fills.append(report)
        return [report]

    def _limit_hit(self, req: OrderRequest, bar: BarEvent) -> bool:
        if req.price is None:
            return False
        if req.side == OrderSide.BUY:
            return bar.low <= req.price
        return bar.high >= req.price

    def _stop_triggered(self, req: OrderRequest, bar: BarEvent) -> bool:
        if req.stop_price is None:
            return False
        if req.side == OrderSide.BUY:
            return bar.high >= req.stop_price
        return bar.low <= req.stop_price

    def _fill_price(self, req: OrderRequest, bar: BarEvent, is_maker: bool) -> float:
        base_price: float
        if req.order_type == OrderType.LIMIT and req.price is not None:
            base_price = req.price
        elif req.order_type == OrderType.STOP_MARKET and req.stop_price is not None:
            base_price = req.stop_price
        else:
            base_price = bar.close
        if is_maker:
            return float(base_price)
        return self._apply_slippage(base_price, req.side)

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        if self.slippage <= 0:
            return float(price)
        adjust = 1.0 + self.slippage if side == OrderSide.BUY else 1.0 - self.slippage
        px = float(price * adjust)
        if self._slippage_jitter_bps and self._slippage_jitter_bps > 0:
            jitter = (random.uniform(-1.0, 1.0)) * (self._slippage_jitter_bps / 1e4)
            px *= (1.0 + jitter)
        return float(px)

    @property
    def open_orders(self) -> Dict[str, _SimulatedOrder]:
        return self._open_orders
