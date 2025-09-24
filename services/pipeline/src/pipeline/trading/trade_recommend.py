from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import os
from ..infra.db import fetch_agent_config
from .portfolio_manager import PortfolioManager  # lightweight dependency
from .live_portfolio_manager import LivePortfolioManager
from ..utils.horizons import minutes_to_horizon
from .optimizer import Signal as RiskSignal, build_adaptive_risk_profile


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
    historical_target_minutes: Optional[List[float]] = None
    historical_stop_minutes: Optional[List[float]] = None


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

    # Agent config / fallbacks
    cfg = fetch_agent_config("TradeRecommend") or {"parameters": {}}
    params = cfg.get("parameters") if isinstance(cfg, dict) else {}
    try:
        k_atr = float(os.getenv("TR_K_ATR", str(params.get("k_atr", 1.5))))
    except Exception:
        k_atr = 1.5
    # Aim R:R target (configurable)
    try:
        rr_target = float(os.getenv("TR_RR_TARGET", str(params.get("rr_target", 1.6))))
    except Exception:
        rr_target = 1.6
    entry = p

    # Baseline volatility scaling (legacy heuristic retained as guard rail)
    vol_scale = min(1.0, max(0.2, (50.0 / max(1.0, atr))))
    if inp.regime == "trend_up" or inp.regime == "trend_down":
        vol_scale = min(1.0, vol_scale * 1.1)
    elif inp.regime == "range":
        vol_scale = max(0.2, vol_scale * 0.8)
    fallback_leverage = float(min(inp.leverage_cap, max(1.0, inp.leverage_cap * vol_scale)))

    forecast_minutes = 240
    if inp.horizon_minutes and inp.horizon_minutes > 0:
        try:
            forecast_minutes = max(1, int(inp.horizon_minutes))
        except Exception:
            forecast_minutes = 240

    projected_rr = abs(target - p) / atr
    projected_rr = max(0.5, min(4.5, float(projected_rr)))

    risk_profile = None
    risk_notes: List[str] = []
    min_hold = 10 if forecast_minutes <= 60 else 20
    try:
        valid_window_baseline = max(10, int(inp.valid_for_minutes))
    except Exception:
        valid_window_baseline = 30

    try:
        risk_signal = RiskSignal(
            name=f"trade:{side.lower()}",
            proba_up=float(inp.proba_up),
            rr=projected_rr,
            atr=atr,
            price=entry,
        )
        risk_profile = build_adaptive_risk_profile(
            risk_signal,
            forecast_horizon=forecast_minutes,
            leverage_cap=inp.leverage_cap,
            historical_target_minutes=inp.historical_target_minutes,
            historical_stop_minutes=inp.historical_stop_minutes,
            min_hold_minutes=min_hold,
            min_valid_minutes=valid_window_baseline,
        )
    except Exception:
        risk_profile = None

    if risk_profile:
        sl_dist, tp_dist = risk_profile.sl_tp_distances(atr)
        rr_expected = float(tp_dist / max(1e-6, sl_dist))
        leverage_candidate = float(min(inp.leverage_cap, max(1.0, inp.leverage_cap * risk_profile.leverage_factor)))
        leverage = min(leverage_candidate, fallback_leverage)
        risk_fraction = max(0.0005, min(0.05, inp.risk_per_trade * risk_profile.risk_multiplier))
        valid_for_minutes = max(1, int(risk_profile.valid_minutes))
        horizon_for_output = risk_profile.hold_minutes
        risk_notes = risk_profile.notes
    else:
        sl_dist = k_atr * atr
        tp_dist = rr_target * sl_dist
        rr_expected = float(tp_dist / max(1e-6, sl_dist))
        leverage = fallback_leverage
        risk_fraction = inp.risk_per_trade
        valid_for_minutes = max(1, int(inp.valid_for_minutes))
        horizon_for_output = inp.horizon_minutes or forecast_minutes
        risk_notes = ["fallback_static_model"]

    sl = entry - sl_dist if side == "LONG" else entry + sl_dist
    tp = entry + tp_dist if side == "LONG" else entry - tp_dist

    risk_amount = inp.account_equity * risk_fraction
    size_qty = float(risk_amount / max(1e-6, abs(entry - sl)))

    # Timing fields
    now_dt = datetime.fromtimestamp(inp.now_ts, tz=timezone.utc) if inp.now_ts else datetime.now(timezone.utc)
    tz = ZoneInfo(inp.tz_name or os.getenv("TIMEZONE", "Asia/Jerusalem"))
    signal_time = now_dt.astimezone(tz)
    valid_until = (now_dt + timedelta(minutes=valid_for_minutes)).astimezone(tz)
    horizon_end = None
    if horizon_for_output and horizon_for_output > 0:
        try:
            horizon_end = (now_dt + timedelta(minutes=int(horizon_for_output))).astimezone(tz)
        except Exception:
            horizon_end = None

    reason_codes = ["directional_bias", f"regime:{inp.regime or 'n/a'}"]
    if risk_profile:
        reason_codes.append("risk_dynamic")
        bucket_note = next((n for n in risk_profile.notes if n.startswith("bucket:")), None)
        if bucket_note:
            reason_codes.append(bucket_note.replace("bucket:", "risk_bucket:"))
    else:
        reason_codes.append("atr_volatility")

    confidence_metric = round(float(abs(inp.proba_up - 0.5) * 2), 2)

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
        "confidence": confidence_metric,
        "reason_codes": reason_codes,
        "invalidation": "пробой SL/сигнал смены направления",
    }
    if risk_profile:
        profile_dict = asdict(risk_profile)
        profile_dict["sl_distance"] = round(sl_dist, 6)
        profile_dict["tp_distance"] = round(tp_dist, 6)
        profile_dict["forecast_minutes"] = forecast_minutes
        try:
            profile_dict["forecast_label"] = minutes_to_horizon(forecast_minutes)
        except Exception:
            profile_dict["forecast_label"] = str(forecast_minutes)
        profile_dict["notes"] = risk_notes
        card["risk_profile"] = profile_dict
    else:
        card["risk_profile"] = {
            "sl_distance": round(sl_dist, 6),
            "tp_distance": round(tp_dist, 6),
            "notes": risk_notes,
            "forecast_minutes": forecast_minutes,
        }
    # Optional portfolio policy overlay (no-op if disabled)
    try:
        use_live = os.getenv("PORTFOLIO_LIVE", "0") in {"1", "true", "True"}
        use_db = os.getenv("PORTFOLIO_ENABLED", "0") in {"1", "true", "True"}
        symbol = os.getenv("EXEC_SYMBOL", "BTC/USDT")
        if use_live:
            lpm = LivePortfolioManager()
            # построим решение, используя ту же политику, что и у PortfolioManager
            from .portfolio_manager import PortfolioManager as _PM

            pm = _PM()
            # временно подменим storage‑доступ через monkey, передадим позицию напрямую
            pos = lpm.get_position(symbol)
            if pos and pos.quantity > 0:
                # создадим простую развязку: если есть позиция — определим решение по направлению
                same = (pos.direction == "long" and side == "LONG") or (pos.direction == "short" and side == "SHORT")
                try:
                    th = float(os.getenv("TR_SCALEIN_CONF_THRES", "0.68"))
                except Exception:
                    th = 0.68
                if same:
                    decision, reason = ("scale_in" if float(card["confidence"]) >= th else "ignore", "confidence_ok" if float(card["confidence"]) >= th else "low_confidence")
                else:
                    decision, reason = ("ignore", "opposite_position_open")
            else:
                decision, reason = ("open", "no_position")
        elif use_db:
            pm = PortfolioManager()
            decision, reason = pm.decide_with_existing(symbol=symbol, new_side=side, confidence=float(card["confidence"]))
        else:
            decision, reason = ("open", "disabled")

        card["portfolio_decision"] = decision
        card["portfolio_reason"] = reason
        if decision == "ignore" and reason == "opposite_position_open":
            card["side"] = "NO-TRADE"
            card["reason_codes"].append("portfolio_block")
        if decision == "scale_in":
            try:
                scale = float(os.getenv("TR_SCALEIN_FRACTION", "0.5"))
            except Exception:
                scale = 0.5
            card["size"]["qty"] = round(card["size"]["qty"] * max(0.0, min(1.0, scale)), 6)
            card["reason_codes"].append("portfolio_scale_in")
    except Exception:
        pass
    return card
