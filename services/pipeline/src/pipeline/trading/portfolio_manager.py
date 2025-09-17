from __future__ import annotations

"""
PortfolioManager — централизованный учёт портфеля, позиций и совокупного риска.

Назначение:
  - агрегировать открытые позиции (direction, qty, avg_price);
  - считать нереализованный PnL и экспозицию по символам;
  - оценивать суммарный риск (сумма потенциальных убытков по SL);
  - предоставлять вспомогательные методы для расчёта размера позиции.

Интеграция:
  - модуль можно использовать из trade_recommend.py для учёта уже открытых
    позиций (масштабирование, запрет реверса и т.п.);
  - при наличии БД — состояния хранятся в таблицах из миграции 038_portfolio.sql;
    иначе доступен InMemory fallback.

По умолчанию — безопасный no-op: если БД недоступна или PORTFOLIO_ENABLED != 1,
менеджер работает в памяти и не влияет на существующие тесты.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os


@dataclass
class Position:
    symbol: str
    direction: str  # "long" | "short"
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0


class PortfolioStorage:
    """Абстракция хранилища состояний (DB или память)."""

    def get_position(self, symbol: str) -> Optional[Position]:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def upsert_position(self, pos: Position) -> None:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def total_equity(self) -> float:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def total_risk(self) -> float:  # pragma: no cover - интерфейс
        raise NotImplementedError


class InMemoryStorage(PortfolioStorage):
    def __init__(self, starting_equity: float = 10_000.0) -> None:
        self._equity = float(starting_equity)
        self._positions: Dict[str, Position] = {}

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def upsert_position(self, pos: Position) -> None:
        self._positions[pos.symbol] = pos

    def total_equity(self) -> float:
        return float(self._equity)

    def total_risk(self) -> float:
        # Пессимистичная оценка риска отсутствует в памяти — вернём 0
        return 0.0


class DBStorage(PortfolioStorage):
    """БД‑хранилище. Требует миграции 038_portfolio.sql.

    Примечание: функции читают/пишут только в свои таблицы и не зависят
    от остального пайплайна. При ошибке — молча деградируют.
    """

    def __init__(self) -> None:
        try:
            from ..infra.db import get_conn  # lazy import
        except Exception:  # pragma: no cover - защитный fallback
            get_conn = None  # type: ignore[assignment]
        self._get_conn = get_conn

    def get_position(self, symbol: str) -> Optional[Position]:
        if not self._get_conn:
            return None
        sql = (
            "SELECT symbol, direction, quantity, avg_price, unrealized_pnl "
            "FROM portfolio_positions WHERE symbol=%s AND closed_at IS NULL ORDER BY updated_at DESC LIMIT 1"
        )
        try:
            with self._get_conn() as conn:  # type: ignore[misc]
                with conn.cursor() as cur:
                    cur.execute(sql, (symbol,))
                    row = cur.fetchone()
                    if not row:
                        return None
                    return Position(
                        symbol=str(row[0]),
                        direction=str(row[1] or ""),
                        quantity=float(row[2] or 0.0),
                        avg_price=float(row[3] or 0.0),
                        unrealized_pnl=float(row[4] or 0.0),
                    )
        except Exception:
            return None

    def upsert_position(self, pos: Position) -> None:
        if not self._get_conn:
            return
        sql = (
            "INSERT INTO portfolio_positions (symbol, direction, quantity, avg_price, unrealized_pnl) "
            "VALUES (%s,%s,%s,%s,%s) "
            "ON CONFLICT (symbol) WHERE closed_at IS NULL "
            "DO UPDATE SET direction=EXCLUDED.direction, quantity=EXCLUDED.quantity, avg_price=EXCLUDED.avg_price, unrealized_pnl=EXCLUDED.unrealized_pnl, updated_at=now()"
        )
        try:
            with self._get_conn() as conn:  # type: ignore[misc]
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        (
                            pos.symbol,
                            pos.direction,
                            float(pos.quantity),
                            float(pos.avg_price),
                            float(pos.unrealized_pnl),
                        ),
                    )
        except Exception:
            return

    def total_equity(self) -> float:
        if not self._get_conn:
            return 0.0
        sql = "SELECT COALESCE(SUM(equity),0.0) FROM portfolio_equity_daily WHERE date=CURRENT_DATE"
        try:
            with self._get_conn() as conn:  # type: ignore[misc]
                with conn.cursor() as cur:
                    cur.execute(sql, ())
                    row = cur.fetchone()
                    return float(row[0] or 0.0)
        except Exception:
            return 0.0

    def total_risk(self) -> float:
        if not self._get_conn:
            return 0.0
        sql = "SELECT COALESCE(SUM(potential_loss),0.0) FROM portfolio_risk_exposure WHERE as_of >= now() - interval '1 hour'"
        try:
            with self._get_conn() as conn:  # type: ignore[misc]
                with conn.cursor() as cur:
                    cur.execute(sql, ())
                    row = cur.fetchone()
                    return float(row[0] or 0.0)
        except Exception:
            return 0.0


class PortfolioManager:
    """Фасад над хранилищем: бизнес‑правила и утилиты расчёта размера позиции."""

    def __init__(self, storage: Optional[PortfolioStorage] = None) -> None:
        use_db = os.getenv("PORTFOLIO_ENABLED", "0") in {"1", "true", "True"}
        self.storage = storage or (DBStorage() if use_db else InMemoryStorage())

    # --- Queries ---------------------------------------------------------
    def get_position(self, symbol: str) -> Optional[Position]:
        return self.storage.get_position(symbol)

    def total_equity(self) -> float:
        return self.storage.total_equity()

    def total_risk(self) -> float:
        return self.storage.total_risk()

    # --- Sizing ----------------------------------------------------------
    def position_size_by_risk(
        self, equity: float, risk_per_trade: float, entry: float, stop: float
    ) -> float:
        """Возвращает размер позиции по правилу фиксированного риска на сделку.

        qty = risk_amount / |entry - stop|
        """
        risk_amount = float(equity) * float(max(0.0, risk_per_trade))
        dist = abs(float(entry) - float(stop))
        if dist <= 1e-9 or risk_amount <= 0:
            return 0.0
        return round(risk_amount / dist, 6)

    # --- Policy helpers --------------------------------------------------
    def decide_with_existing(
        self,
        symbol: str,
        new_side: str,  # "LONG" | "SHORT"
        confidence: float,
    ) -> Tuple[str, str]:
        """Принять решение с учётом открытой позиции.

        Возвращает (decision, reason):
            - "open"       — нет позиции, можно открыть;
            - "scale_in"   — позиция в том же направлении, можно добрать;
            - "ignore"     — есть позиция в противоположном направлении;
        Порог уверенности для добора управляется TR_SCALEIN_CONF_THRES (0.68 по умолч.).
        """
        pos = self.get_position(symbol)
        if not pos or pos.quantity <= 0:
            return "open", "no_position"
        same_dir = (pos.direction == "long" and new_side == "LONG") or (
            pos.direction == "short" and new_side == "SHORT"
        )
        if same_dir:
            try:
                th = float(os.getenv("TR_SCALEIN_CONF_THRES", "0.68"))
            except Exception:
                th = 0.68
            return ("scale_in", "confidence_ok") if confidence >= th else ("ignore", "low_confidence")
        # Противоположная позиция запрещает открытие новой
        return "ignore", "opposite_position_open"
