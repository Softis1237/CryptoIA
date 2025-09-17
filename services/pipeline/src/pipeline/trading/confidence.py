from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ..infra.db import fetch_agent_config


@dataclass
class ConfidenceResult:
    value: float
    factors: Dict[str, float]


def _cfg() -> Dict[str, Any]:
    try:
        cfg = fetch_agent_config("ConfidenceAggregator") or {"parameters": {}}
        return cfg.get("parameters") or {}
    except Exception:
        return {}


def compute(base_conf: float, context: Dict[str, Any]) -> ConfidenceResult:
    """Aggregate confidence from base and contextual factors.

    Expects context keys: trigger_type, ta_sentiment, regime_label, side.
    Optionally: pattern_hit(bool), deriv_signal(bool), news_impact(float), alert_priority(str).
    """
    p = _cfg()
    # weights with defaults
    w = {
        "trigger": float(p.get("w_trigger", 0.08)),
        "pattern": float(p.get("w_pattern", 0.10)),
        "deriv": float(p.get("w_deriv", 0.08)),
        "ta": float(p.get("w_ta", 0.08)),
        "regime": float(p.get("w_regime", 0.05)),
        "news": float(p.get("w_news", 0.05)),
        "alert": float(p.get("w_alert", 0.05)),
        "smc": float(p.get("w_smc", 0.10)),
        "whale": float(p.get("w_whale", 0.08)),
        "alpha": float(p.get("w_alpha", 0.10)),
    }
    side = str(context.get("side") or "").upper()
    trig = str(context.get("trigger_type") or "").upper()
    ta_sent = str(context.get("ta_sentiment") or "neutral")
    regime = str(context.get("regime_label") or "range")
    pattern_hit = bool(context.get("pattern_hit", False))
    deriv_sig = bool(context.get("deriv_signal", False))
    news_imp = float(context.get("news_impact", 0.0) or 0.0)
    alert_prio = str(context.get("alert_priority", "low"))
    smc_status = str(context.get("smc_status", "")).upper()
    whale_status = str(context.get("whale_status", "")).upper()
    alpha_support = context.get("alpha_support", False)
    alpha_score = context.get("alpha_score", None)

    factors: Dict[str, float] = {}

    # Trigger-based bonus
    if trig in {"MOMENTUM", "ATR_SPIKE", "VOL_SPIKE", "DELTA_SPIKE"}:
        factors["trigger"] = w["trigger"]
    elif trig.startswith("PATTERN_") or trig.startswith("DERIV_") or trig in {"L2_IMBALANCE", "L2_WALL", "NEWS"}:
        factors["trigger"] = w["trigger"] * 0.8

    # Pattern/derivatives explicit
    if pattern_hit:
        factors["pattern"] = w["pattern"]
    if deriv_sig:
        factors["deriv"] = w["deriv"]

    # TA alignment
    if (side == "LONG" and ta_sent.startswith("bull")) or (side == "SHORT" and ta_sent.startswith("bear")):
        factors["ta"] = w["ta"]

    # Regime alignment
    if (side == "LONG" and regime == "trend_up") or (side == "SHORT" and regime == "trend_down"):
        factors["regime"] = w["regime"]

    # News impact scaled
    if news_imp > 0:
        factors["news"] = min(w["news"], news_imp * w["news"])  # cap

    # Alert priority bonus
    if alert_prio in {"high", "critical"}:
        factors["alert"] = w["alert"] * (1.0 if alert_prio == "high" else 1.4)

    # Strategic agents alignment (dynamic weight by regime)
    trend_mult = 1.2 if regime in {"trend_up", "trend_down"} else (0.9 if regime == "range" else 1.0)
    if (side == "LONG" and smc_status.startswith("SMC_BULLISH")) or (side == "SHORT" and smc_status.startswith("SMC_BEARISH")):
        factors["smc"] = w["smc"] * trend_mult
    if (side == "LONG" and whale_status.endswith("BULLISH")) or (side == "SHORT" and whale_status.endswith("BEARISH")):
        factors["whale"] = w["whale"] * trend_mult
    # Alpha strategies support
    try:
        if isinstance(alpha_support, bool) and alpha_support:
            factors["alpha"] = w["alpha"]
        elif alpha_score is not None:
            s = float(alpha_score)
            if s > 0:
                factors["alpha"] = max(0.0, min(w["alpha"], s * w["alpha"]))
    except Exception:
        pass

    out = float(base_conf) + sum(factors.values())
    out = max(0.0, min(1.0, out))
    return ConfidenceResult(value=round(out, 2), factors={k: round(v, 3) for k, v in factors.items()})
