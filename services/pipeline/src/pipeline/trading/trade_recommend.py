from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import os
from ..infra.db import fetch_agent_config
from .portfolio_manager import PortfolioManager  # lightweight dependency


@dataclass
class TradeRecommendInput:
    current_price: float
    y_hat_4h: float
    interval_low: float
    interval_high: float
    proba_up: float
    atr: float
    regime: Optional[str] = None
    volatility: Optional[float] = None
    account_equity: float = 1000.0
    risk_per_trade: float = 0.005
    leverage_cap: float = 25.0
    # Timing/options
    now_ts: Optional[int] = None  # seconds since epoch (UTC)
    tz_name: Optional[str] = None  # e.g. Asia/Jerusalem
    valid_for_minutes: int = 90
    horizon_minutes: Optional[int] = None


def run(inp: TradeRecommendInput) -> Dict:
    p = inp.current_price
    target = inp.y_hat_4h
    atr = max(1e-6, inp.atr)

    # Direction
    dir_is_long = (target > p and inp.proba_up >= 0.55)
    dir_is_short = (target < p and inp.proba_up <= 0.45)
    if not (dir_is_long or dir_is_short):
        return {
            "signal_time": None,
            "valid_until": None,
            "horizon_ends_at": None,
            "side": "NO-TRADE",
            "reason_codes": ["low_confidence_or_conflicting"],
        }

    side = "LONG" if dir_is_long else "SHORT"

    # Volatility-aware SL/TP distances (configurable)
    cfg = fetch_agent_config("TradeRecommend") or {"parameters": {}}
    params = cfg.get("parameters") if isinstance(cfg, dict) else {}
    try:
        k_atr = float(os.getenv("TR_K_ATR", str(params.get("k_atr", 1.5))))
    except Exception:
        k_atr = 1.5
    sl_dist = k_atr * atr
    # Aim R:R target (configurable)
    try:
        rr_target = float(os.getenv("TR_RR_TARGET", str(params.get("rr_target", 1.6))))
    except Exception:
        rr_target = 1.6
    tp_dist = rr_target * sl_dist

    entry = p
    sl = entry - sl_dist if side == "LONG" else entry + sl_dist
    tp = entry + tp_dist if side == "LONG" else entry - tp_dist

    # Leverage scaling by volatility (higher vol -> lower leverage)
    vol_scale = min(1.0, max(0.2, (50.0 / max(1.0, atr))))  # heuristic

    # Regime adjustments: in strong trend allow slightly higher scale; in range reduce
    if inp.regime == "trend_up" or inp.regime == "trend_down":
        vol_scale = min(1.0, vol_scale * 1.1)
    elif inp.regime == "range":
        vol_scale = max(0.2, vol_scale * 0.8)
    leverage = float(min(inp.leverage_cap, max(1.0, inp.leverage_cap * vol_scale)))

    # Position size by risk per trade
    risk_amount = inp.account_equity * inp.risk_per_trade
    size_qty = float(risk_amount / max(1e-6, abs(entry - sl)))

    rr_expected = float(abs(tp - entry) / max(1e-6, abs(entry - sl)))

    # Timing fields
    now_dt = datetime.fromtimestamp(inp.now_ts, tz=timezone.utc) if inp.now_ts else datetime.now(timezone.utc)
    tz = ZoneInfo(inp.tz_name or os.getenv("TIMEZONE", "Asia/Jerusalem"))
    signal_time = now_dt.astimezone(tz)
    valid_until = (now_dt + timedelta(minutes=max(1, int(inp.valid_for_minutes)))).astimezone(tz)
    horizon_end = None
    if inp.horizon_minutes and inp.horizon_minutes > 0:
        horizon_end = (now_dt + timedelta(minutes=int(inp.horizon_minutes))).astimezone(tz)

    card = {
        "signal_time": signal_time.isoformat(timespec="seconds"),
        "valid_until": valid_until.isoformat(timespec="seconds"),
        "horizon_ends_at": horizon_end.isoformat(timespec="seconds") if horizon_end else None,
        "side": side,
        "entry": {"type": "market", "zone": [entry, entry]},
        "leverage": round(leverage, 2),
        "size": {"qty": round(size_qty, 6), "unit": "BTC"},
        "stop_loss": round(sl, 2),
        "take_profit": round(tp, 2),
        "rr_expected": round(rr_expected, 2),
        "confidence": round(float(abs(inp.proba_up - 0.5) * 2), 2),
        "reason_codes": ["directional_bias", "atr_volatility", f"regime:{inp.regime or 'n/a'}"],
        "invalidation": "пробой SL/сигнал смены направления",
    }
    # Optional portfolio policy overlay (no-op if disabled)
    try:
        if os.getenv("PORTFOLIO_ENABLED", "0") in {"1", "true", "True"}:
            pm = PortfolioManager()
            decision, reason = pm.decide_with_existing(
                symbol=os.getenv("EXEC_SYMBOL", "BTC/USDT"),
                new_side=side,
                confidence=float(card["confidence"]),
            )
            card["portfolio_decision"] = decision
            card["portfolio_reason"] = reason
            # if decision is "ignore" due to opposite position — conservatively convert to NO-TRADE
            if decision == "ignore" and reason == "opposite_position_open":
                card["side"] = "NO-TRADE"
                card["reason_codes"].append("portfolio_block")
            # if scale-in — recompute qty proportionally (half of base risk by default)
            if decision == "scale_in":
                try:
                    scale = float(os.getenv("TR_SCALEIN_FRACTION", "0.5"))
                except Exception:
                    scale = 0.5
                card["size"]["qty"] = round(card["size"]["qty"] * max(0.0, min(1.0, scale)), 6)
                card["reason_codes"].append("portfolio_scale_in")
    except Exception:
        # не мешаем основной логике при любых проблемах
        pass
    return card
