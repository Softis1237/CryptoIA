from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

from .exchange_simulator import ExchangeSimulator
from .portfolio import Portfolio, build_portfolio
from .report import BacktestReport, ReportBuilder
from .strategies import load_strategy
from .types import (
    BacktestConfig,
    BarEvent,
    DataFeed,
    OrderRequest,
    OrderType,
    PortfolioState,
    Strategy,
)


@dataclass
class DataFrameFeed(DataFeed):
    """Итератор по OHLC данным DataFrame."""

    df: pd.DataFrame
    symbol: str
    timezone: Optional[str] = None

    def __iter__(self) -> Iterable[BarEvent]:
        frame = self.df.copy()
        ts = pd.to_datetime(frame["timestamp"], utc=False)
        if self.timezone:
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize(self.timezone)
            else:
                ts = ts.dt.tz_convert(self.timezone)
        frame["timestamp"] = ts
        frame = frame.sort_values("timestamp")
        for row in frame.to_dict(orient="records"):
            event = BarEvent(
                timestamp=pd.Timestamp(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
                extras={k: v for k, v in row.items() if k not in {"timestamp", "open", "high", "low", "close", "volume"}},
            )
            event.extras.setdefault("symbol", self.symbol)
            yield event


class VectorBacktester:
    """Основной класс векторного бэктестера."""

    def __init__(
        self,
        config: BacktestConfig,
        feed: DataFeed,
        strategy: Strategy,
        exchange: Optional[ExchangeSimulator] = None,
    ) -> None:
        self.config = config
        self.feed = feed
        self.strategy = strategy
        self.exchange = exchange or ExchangeSimulator(
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
            slippage=config.slippage,
            volume_limit=config.volume_limit,
        )
        self.portfolio: Portfolio = build_portfolio(config.starting_cash)
        self.report_builder = ReportBuilder()
        self._history: List[BarEvent] = []
        self._last_trade_idx = 0

    def run(self) -> BacktestReport:
        for bar in self.feed:
            self.exchange._current_bar = bar
            pre_fills = self.exchange.process_bar(bar)
            self._apply_fills(pre_fills)

            symbol = bar.extras.get("symbol") if bar.extras else None
            self.portfolio.update_mark_price(symbol or self.config.symbol, bar.close)
            snapshot_pre = self.portfolio.mark_to_market(bar.timestamp, store=False)
            state = PortfolioState.from_snapshot(snapshot_pre)
            state.metadata.setdefault("config_symbol", self.config.symbol)

            orders = self.strategy.on_bar(bar, tuple(self._history), state)
            for order in orders:
                order = self._prepare_order(order, bar)
                sim_order = self.exchange.submit(order, bar.timestamp)
                self._apply_fills(sim_order.fills)

            final_snapshot = self.portfolio.mark_to_market(bar.timestamp, store=True)
            final_snapshot.metadata.setdefault("config_symbol", self.config.symbol)
            self.report_builder.add_snapshot(final_snapshot)
            self._capture_new_trades()
            self._history.append(bar)

        self.strategy.finalize()
        return self.report_builder.build(
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=self._infer_periods_per_year(),
        )

    # ------------------------------------------------------------------
    def _prepare_order(self, order: OrderRequest, bar: BarEvent) -> OrderRequest:
        order.submitted_at = bar.timestamp
        if not order.symbol:
            order.symbol = self.config.symbol
        if order.order_type not in {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_MARKET}:
            raise ValueError(f"Неподдерживаемый тип ордера {order.order_type}")
        return order

    def _apply_fills(self, fills: Iterable) -> None:
        if not fills:
            return
        for fill in fills:
            self.portfolio.process_fill(fill)

    def _capture_new_trades(self) -> None:
        trades = self.portfolio.get_trades()
        if self._last_trade_idx < len(trades):
            self.report_builder.extend_trades(trades[self._last_trade_idx :])
            self._last_trade_idx = len(trades)

    def _infer_periods_per_year(self) -> int:
        try:
            delta = pd.to_timedelta(self.config.report_frequency)
            if delta <= pd.Timedelta(0):
                raise ValueError
            periods = int(round(pd.Timedelta(days=365) / delta))
            return max(1, periods)
        except Exception:
            return 365


def run_from_dataframe(
    df: pd.DataFrame,
    config: BacktestConfig,
    strategy_path: str,
    strategy_params: Optional[dict] = None,
) -> BacktestReport:
    feed = DataFrameFeed(df=df, symbol=config.symbol, timezone=config.data_timezone)
    strategy = load_strategy(strategy_path, strategy_params)
    bt = VectorBacktester(config=config, feed=feed, strategy=strategy)
    return bt.run()
