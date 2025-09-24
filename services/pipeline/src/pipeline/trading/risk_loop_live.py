from __future__ import annotations

import os
import time
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from .exchange import ExchangeClient
from ..infra.db import (
    fetch_prediction_for_run,
    get_conn,
    insert_orchestrator_event,
)

try:  # pragma: no cover - telegram optional
    from ..telegram_bot.publisher import publish_message, publish_message_to  # type: ignore
except Exception:  # pragma: no cover
    publish_message = None  # type: ignore
    publish_message_to = None  # type: ignore


@dataclass
class TrailConfig:
    trail_pct: float = 0.01  # 1%
    improve_threshold_pct: float = 0.003  # only move if improves by 0.3%
    interval_s: int = 60


@dataclass
class GuardianCandidate:
    symbol: str
    side: str  # LONG | SHORT
    entry: float
    sl: float
    tp: float
    qty: float
    opened_at: datetime
    current_price: float
    source: str
    run_id: Optional[str] = None
    risk_profile: Dict[str, Any] | None = None
    base_probability: Optional[float] = None


def _trail_stop_for_long(entry: float, high: float, trail_pct: float) -> float:
    return float(high * (1.0 - trail_pct))


def _trail_stop_for_short(entry: float, low: float, trail_pct: float) -> float:
    return float(low * (1.0 + trail_pct))


class PositionGuardianAgent:
    """Re-assess open positions and alert when probability drops."""

    def __init__(
        self,
        symbol: Optional[str] = None,
        prob_threshold: Optional[float] = None,
        price_fetcher: Optional[callable] = None,
    ) -> None:
        self.symbol = symbol or os.getenv("EXEC_SYMBOL", "BTC/USDT")
        try:
            self.prob_threshold = float(
                prob_threshold
                if prob_threshold is not None
                else os.getenv("GUARDIAN_PROB_THRESHOLD", 0.3)
            )
        except Exception:
            self.prob_threshold = 0.3
        self._price_fetcher = price_fetcher or self._default_price_fetcher
        self._exchange: Optional[ExchangeClient] = None

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        history = self._fetch_history_stats()
        candidates = self._collect_candidates()
        alerts: List[Dict[str, Any]] = []
        for cand in candidates:
            assessment = self._assess_candidate(cand, history)
            if assessment["should_alert"]:
                alerts.append(assessment)
                self._notify(assessment)
                self._log_event(assessment)
        return {"checked": len(candidates), "alerts": alerts}

    # ------------------------------------------------------------------
    def _default_price_fetcher(self, symbol: str) -> float:
        if self._exchange is None:
            self._exchange = ExchangeClient()
        try:
            return float(self._exchange.get_last_price(symbol))
        except Exception as exc:  # pragma: no cover - network dependent
            logger.debug(f"PositionGuardian: price fetch failed: {exc}")
            return 0.0

    def _collect_candidates(self) -> List[GuardianCandidate]:
        out: List[GuardianCandidate] = []
        symbol = self.symbol
        # Paper positions with run mapping
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT p.pos_id, p.side, p.entry, p.sl, p.tp, p.qty, p.opened_at, "
                        "       p.meta_json->>'run_id' as run_id, t.times_json "
                        "FROM paper_positions p "
                        "JOIN trades_suggestions t ON (t.run_id = p.meta_json->>'run_id') "
                        "WHERE p.status='OPEN'"
                    )
                    rows = cur.fetchall() or []
        except Exception as exc:  # pragma: no cover - db optional
            logger.debug(f"PositionGuardian: fetch paper positions failed: {exc}")
            rows = []
        price = self._price_fetcher(symbol) if rows else 0.0
        for pos_id, side, entry, sl, tp, qty, opened_at, run_id, times_json in rows:
            try:
                current_price = price if price > 0 else self._price_fetcher(symbol)
                base_pred = (
                    fetch_prediction_for_run(run_id, "4h") if run_id else None
                )
                risk_profile = None
                if isinstance(times_json, dict):
                    risk_profile = times_json.get("risk_profile")
                out.append(
                    GuardianCandidate(
                        symbol=symbol,
                        side=str(side).upper(),
                        entry=float(entry),
                        sl=float(sl),
                        tp=float(tp),
                        qty=float(qty),
                        opened_at=opened_at.replace(tzinfo=timezone.utc)
                        if opened_at.tzinfo is None
                        else opened_at,
                        current_price=float(current_price),
                        source="paper",
                        run_id=str(run_id) if run_id else None,
                        risk_profile=risk_profile if isinstance(risk_profile, dict) else None,
                        base_probability=float(base_pred["proba_up"]) if base_pred else None,
                    )
                )
            except Exception as exc:  # pragma: no cover
                logger.debug(f"PositionGuardian: build candidate failed: {exc}")
                continue
        return out

    def _assess_candidate(self, cand: GuardianCandidate, history: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        success_probability, proba_up = self._estimate_probability(cand, history)
        should_alert = success_probability < self.prob_threshold
        return {
            "run_id": cand.run_id,
            "symbol": cand.symbol,
            "side": cand.side,
            "current_price": cand.current_price,
            "success_probability": round(success_probability, 3),
            "proba_up": round(proba_up, 3),
            "should_alert": should_alert,
            "source": cand.source,
        }

    def _estimate_probability(self, cand: GuardianCandidate, history: Dict[str, Dict[str, Any]]) -> tuple[float, float]:
        base = cand.base_probability if cand.base_probability is not None else 0.55
        history_stats = history.get(cand.side, {})
        success_rate = history_stats.get("success_rate")
        if isinstance(success_rate, (int, float)):
            base = max(0.1, min(0.9, base * (0.6 + 0.4 * max(0.0, min(1.0, success_rate)))))
        if cand.side == "SHORT":
            base = 1.0 - base

        range_total = max(1e-6, abs(cand.tp - cand.sl))
        price_delta = cand.current_price - cand.entry if cand.side == "LONG" else cand.entry - cand.current_price
        price_factor = 0.5 + 0.5 * math.tanh((price_delta / range_total) * 2.5)

        hold_minutes = None
        if cand.risk_profile and "hold_minutes" in cand.risk_profile:
            try:
                hold_minutes = float(cand.risk_profile.get("hold_minutes"))
            except Exception:
                hold_minutes = None
        elapsed_minutes = max(
            1.0,
            (datetime.now(timezone.utc) - cand.opened_at).total_seconds() / 60.0,
        )
        success_minutes = history_stats.get("success_minutes") or []
        failure_minutes = history_stats.get("failure_minutes") or []
        if hold_minutes:
            ratio = elapsed_minutes / max(1.0, hold_minutes)
            if ratio <= 1.0:
                time_factor = 0.6 + 0.4 * (1.0 - ratio)
            else:
                time_factor = max(0.2, 0.6 - 0.3 * (ratio - 1.0))
        else:
            time_factor = 0.5

        if success_minutes:
            from statistics import median

            median_success = median(success_minutes)
            if elapsed_minutes <= median_success:
                time_factor = min(1.0, time_factor + 0.15)
            else:
                over = (elapsed_minutes - median_success) / max(1.0, median_success)
                time_factor = max(0.05, time_factor - min(0.25, over * 0.2))
        if failure_minutes:
            from statistics import median

            median_failure = median(failure_minutes)
            if elapsed_minutes >= median_failure:
                over = (elapsed_minutes - median_failure) / max(1.0, median_failure)
                time_factor = max(0.05, time_factor - min(0.3, over * 0.25))

        success = (base * 0.6) + (price_factor * 0.25) + (time_factor * 0.15)
        success = max(0.0, min(1.0, success))
        proba_up = success if cand.side == "LONG" else 1.0 - success
        return success, proba_up

    def _fetch_history_stats(self) -> Dict[str, Dict[str, Any]]:
        sql = (
            "SELECT p.side, tr.reason, EXTRACT(EPOCH FROM (tr.ts - p.opened_at))/60.0 AS minutes "
            "FROM paper_positions p "
            "JOIN paper_trades tr ON tr.pos_id = p.pos_id "
            "WHERE p.status='CLOSED' AND tr.side='CLOSE' "
            "AND tr.ts >= now() - interval '30 days' "
            "ORDER BY tr.ts DESC LIMIT 400"
        )
        def _bucket() -> Dict[str, Any]:
            return {
                "success_minutes": [],
                "failure_minutes": [],
                "success_count": 0,
                "failure_count": 0,
                "total": 0,
            }

        stats: Dict[str, Dict[str, Any]] = {
            "LONG": _bucket(),
            "SHORT": _bucket(),
        }
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    for side, reason, minutes in cur.fetchall() or []:
                        side_key = str(side).upper()
                        if side_key not in stats or minutes is None:
                            continue
                        try:
                            minutes_val = float(minutes)
                        except Exception:
                            continue
                        bucket = stats[side_key]
                        bucket["total"] += 1
                        if str(reason) == "take_profit":
                            bucket["success_minutes"].append(minutes_val)
                            bucket["success_count"] += 1
                        elif str(reason) == "stop_loss":
                            bucket["failure_minutes"].append(minutes_val)
                            bucket["failure_count"] += 1
        except Exception as exc:  # pragma: no cover
            logger.debug(f"PositionGuardian: history fetch failed: {exc}")
        for side_key, payload in stats.items():
            total = payload.get("total", 0) or 0
            if total:
                payload["success_rate"] = payload.get("success_count", 0) / total
                payload["failure_rate"] = payload.get("failure_count", 0) / total
            else:
                payload["success_rate"] = None
                payload["failure_rate"] = None
        return stats

    def _notify(self, assessment: Dict[str, Any]) -> None:
        if not assessment.get("should_alert"):
            return
        message = (
            "Внимание! Вероятность успеха сделки по {symbol} снизилась до {prob:.0f}%"
            " (сторона {side}). Рекомендуется закрыть позицию во избежание убытков."
        ).format(
            symbol=assessment.get("symbol"),
            prob=assessment.get("success_probability", 0.0) * 100,
            side=assessment.get("side"),
        )
        chat_id = os.getenv("TELEGRAM_ALERT_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
        try:
            if publish_message_to and chat_id:
                publish_message_to(chat_id, message)
            elif publish_message:
                publish_message(message)
        except Exception as exc:  # pragma: no cover - telegram optional
            logger.debug(f"PositionGuardian: telegram notify failed: {exc}")

    def _log_event(self, assessment: Dict[str, Any]) -> None:
        try:
            insert_orchestrator_event("position_guardian_alert", assessment)
        except Exception as exc:  # pragma: no cover - db optional
            logger.debug(f"PositionGuardian: failed to log event: {exc}")


def position_guardian_once(symbol: Optional[str] = None) -> Dict[str, Any]:
    agent = PositionGuardianAgent(symbol=symbol)
    return agent.run()


def guardian_loop(interval_s: int = 900) -> None:
    agent = PositionGuardianAgent()
    interval_s = max(60, int(interval_s))
    while True:
        try:
            res = agent.run()
            logger.debug(f"PositionGuardian: {res}")
        except Exception as exc:  # pragma: no cover
            logger.exception(f"PositionGuardian loop error: {exc}")
        time.sleep(interval_s)

def risk_loop(symbol: str = os.getenv("EXEC_SYMBOL", "BTC/USDT"), config: Optional[TrailConfig] = None) -> None:
    cfg = config or TrailConfig(
        trail_pct=float(os.getenv("RISK_TRAIL_PCT", "0.01")),
        improve_threshold_pct=float(os.getenv("RISK_IMPROVE_PCT", "0.003")),
        interval_s=int(os.getenv("RISK_LOOP_INTERVAL", "60")),
    )
    ex = ExchangeClient()
    dry = os.getenv("DRY_RUN", "1") in {"1", "true", "True"}
    high = None
    low = None
    while True:
        try:
            px = ex.get_last_price(symbol)
            high = px if high is None else max(high, px)
            low = px if low is None else min(low, px)
            # NOTE: For simplicity we do not introspect positions here; in practice fetch open positions and adjust stop orders.
            # This loop demonstrates how to compute trailing stops and would need exchange-specific mapping for live orders.
            logger.debug(f"risk_loop {symbol}: px={px:.2f} hi={high:.2f} lo={low:.2f}")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"risk_loop error: {e}")
        time.sleep(cfg.interval_s)


def main():
    mode = os.getenv("RISK_LOOP_MODE", "trail").lower()
    if mode == "guardian":
        try:
            interval_min = float(os.getenv("GUARDIAN_INTERVAL_MIN", "15"))
        except Exception:
            interval_min = 15.0
        guardian_loop(interval_s=int(interval_min * 60))
    else:
        risk_loop()


if __name__ == "__main__":
    main()
