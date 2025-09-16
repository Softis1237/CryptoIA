from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional

import pandas as pd


class OrderType(str, Enum):
    """Тип ордера в симуляторе биржи."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"


class OrderSide(str, Enum):
    """Направление сделки."""

    BUY = "buy"
    SELL = "sell"


@dataclass(slots=True)
class OrderRequest:
    """Запрос на размещение ордера в бэктестере."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    submitted_at: Optional[pd.Timestamp] = None
    time_in_force: str = "GTC"
    client_tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionReport:
    """Факт исполнения ордера в ходе симуляции."""

    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    fee: float
    timestamp: pd.Timestamp
    is_partial: bool
    remaining: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BarEvent:
    """OHLCV-бар, поступающий в бэктестер."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: pd.Series) -> "BarEvent":
        ts = row.get("timestamp")
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        return cls(
            timestamp=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
            extras={k: row[k] for k in row.index if k not in {"timestamp", "open", "high", "low", "close", "volume"}},
        )


@dataclass(slots=True)
class TradeRecord:
    """Закрытая сделка, зафиксированная отчётчиком."""

    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    max_adverse_excursion: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestConfig:
    """Конфигурация запуска векторного бэктестера."""

    symbol: str
    starting_cash: float = 10_000.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage: float = 0.0005
    volume_limit: float = 0.25
    max_leverage: float = 3.0
    warmup_bars: int = 0
    data_timezone: Optional[str] = None
    report_frequency: str = "1D"
    risk_free_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)


class DataFeed(Iterable[BarEvent]):
    """Протокол источника OHLC-баров."""

    def __iter__(self) -> Iterable[BarEvent]:  # pragma: no cover - протокол
        raise NotImplementedError


class Strategy:
    """Протокол торговой стратегии."""

    def on_bar(self, bar: BarEvent, history: Iterable[BarEvent], portfolio_state: "PortfolioState") -> list[OrderRequest]:
        raise NotImplementedError

    def finalize(self) -> None:
        """Выполняет финальные действия после завершения бэктеста."""


@dataclass
class PortfolioSnapshot:
    timestamp: pd.Timestamp
    equity: float
    cash: float
    positions_value: float
    leverage: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioState:
    """Агрегированное состояние портфеля, передаваемое стратегиям."""

    cash: float
    equity: float
    positions_value: float
    leverage: float
    exposure: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, snap: PortfolioSnapshot) -> "PortfolioState":
        return cls(
            cash=snap.cash,
            equity=snap.equity,
            positions_value=snap.positions_value,
            leverage=snap.leverage,
            exposure=snap.metadata.get("exposure", {}),
            metadata={k: v for k, v in snap.metadata.items() if k != "exposure"},
        )
