from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .types import PortfolioSnapshot, TradeRecord


@dataclass
class BacktestReport:
    """Содержит результаты прогона бэктеста."""

    equity_curve: pd.DataFrame
    trades: List[TradeRecord]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "equity_curve": self.equity_curve.to_dict(orient="records"),
            "metrics": self.metrics,
            "trades": [
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                    "metadata": t.metadata,
                }
                for t in self.trades
            ],
        }


class ReportBuilder:
    """Накапливает информацию по ходу бэктеста и рассчитывает метрики."""

    def __init__(self) -> None:
        self.snapshots: List[PortfolioSnapshot] = []
        self.trades: List[TradeRecord] = []

    def add_snapshot(self, snap: PortfolioSnapshot) -> None:
        self.snapshots.append(snap)

    def extend_trades(self, trades: List[TradeRecord]) -> None:
        if trades:
            self.trades.extend(trades)

    def build(self, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> BacktestReport:
        if not self.snapshots:
            raise RuntimeError("Недостаточно данных для формирования отчёта")
        eq_df = pd.DataFrame(
            {
                "timestamp": [snap.timestamp for snap in self.snapshots],
                "equity": [snap.equity for snap in self.snapshots],
                "cash": [snap.cash for snap in self.snapshots],
                "positions_value": [snap.positions_value for snap in self.snapshots],
                "leverage": [snap.leverage for snap in self.snapshots],
            }
        ).set_index("timestamp")
        eq_df.sort_index(inplace=True)
        returns = eq_df["equity"].pct_change().dropna()
        metrics = self._compute_metrics(eq_df, returns, risk_free_rate, periods_per_year)
        return BacktestReport(equity_curve=eq_df.reset_index(), trades=list(self.trades), metrics=metrics)

    def _compute_metrics(
        self,
        equity_curve: pd.DataFrame,
        returns: pd.Series,
        risk_free_rate: float,
        periods_per_year: int,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["total_return"] = float(equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1.0)
        metrics["max_drawdown"] = float(self._max_drawdown(equity_curve["equity"]))
        metrics.update(self._trade_metrics(self.trades))
        if not returns.empty:
            excess = returns - risk_free_rate / periods_per_year
            metrics["sharpe_ratio"] = float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)) if excess.std(ddof=1) > 0 else 0.0
        else:
            metrics["sharpe_ratio"] = 0.0
        return metrics

    def _trade_metrics(self, trades: List[TradeRecord]) -> Dict[str, float]:
        if not trades:
            return {"win_rate": 0.0, "profit_factor": 0.0, "avg_trade": 0.0}
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        profit_factor = sum(wins) / sum(losses) if losses else float("inf")
        avg_trade = sum(t.pnl for t in trades) / len(trades)
        return {
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_trade": float(avg_trade),
        }

    def _max_drawdown(self, equity: pd.Series) -> float:
        roll_max = equity.cummax()
        drawdown = equity / roll_max - 1.0
        return float(drawdown.min())
