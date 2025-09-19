from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .queue import get_key as _get, setex as _setex, incrby as _incr
from ..infra.db import fetch_agent_config


@dataclass
class AlertPriority:
    label: str  # one of low/medium/high/critical
    score: float
    evidence: Dict[str, float]


def _weights() -> Dict[str, float]:
    """Return trigger weights, overrideable via agent_configurations for agent 'AlertPriority'."""
    try:
        cfg = fetch_agent_config("AlertPriority") or {"parameters": {}}
        params = cfg.get("parameters") or {}
    except Exception:
        params = {}
    default = {
        "VOL_SPIKE": 1.0,
        "DELTA_SPIKE": 1.0,
        "ATR_SPIKE": 1.2,
        "MOMENTUM": 1.3,
        "NEWS": 1.4,
        "L2_WALL": 0.8,
        "L2_IMBALANCE": 1.0,
        "PATTERN_BULL_ENGULF": 1.2,
        "PATTERN_BEAR_ENGULF": 1.2,
        "PATTERN_HAMMER": 1.0,
        "PATTERN_SHOOTING_STAR": 1.0,
        "PATTERN_DOJI": 0.8,
        "PATTERN_HANGING_MAN": 1.0,
        "PATTERN_INVERTED_HAMMER": 1.0,
        "PATTERN_MORNING_STAR": 1.2,
        "PATTERN_EVENING_STAR": 1.2,
        "DERIV_OI_JUMP": 1.2,
        "DERIV_FUNDING": 1.1,
        "BREAKOUT_UP": 1.4,
        "BREAKOUT_DOWN": 1.4,
        "BREAKOUT_SMC_UP": 1.5,
        "BREAKOUT_SMC_DOWN": 1.5,
    }
    try:
        override = params.get("weights") or {}
        for k, v in override.items():
            default[str(k).upper()] = float(v)
    except Exception:
        pass
    return default


def mark_and_score(current_type: str, window_sec: int = 180) -> AlertPriority:
    """Mark current trigger in Redis window and compute composite priority score.

    We accumulate counts for each trigger type within TTL=window_sec and sum weighted counts.
    """
    t = str(current_type).upper()
    key = f"rt:win:{t}"
    try:
        # increment and set TTL for the sliding window key
        _incr(key, 1.0)
        _setex(key, window_sec, _get(key) or "1")
    except Exception:
        pass
    w = _weights()
    evidence: Dict[str, float] = {}
    score = 0.0
    # read known keys and accumulate
    for k, wk in w.items():
        try:
            v_raw = _get(f"rt:win:{k}")
            if v_raw is None:
                continue
            v = float(v_raw)
            if v <= 0:
                continue
            s = v * float(wk)
            evidence[k] = s
            score += s
        except Exception:
            continue
    # normalize and map to label with configurable thresholds
    try:
        cfg = fetch_agent_config("AlertPriority") or {"parameters": {}}
        params = cfg.get("parameters") or {}
    except Exception:
        params = {}
    thr = params.get("thresholds") or {}
    t_low = float(thr.get("low", 0.0))
    t_med = float(thr.get("medium", 1.0))
    t_high = float(thr.get("high", 2.0))
    t_crit = float(thr.get("critical", 3.0))
    if score >= t_crit:
        label = "critical"
    elif score >= t_high:
        label = "high"
    elif score >= t_med:
        label = "medium"
    else:
        label = "low"

    # Strengthen critical: require combos if configured
    if label == "critical":
        combos = params.get("critical_combos") or [
            ["MOMENTUM", "VOL_SPIKE"],
            ["MOMENTUM", "PATTERN_BULL_ENGULF"],
            ["MOMENTUM", "PATTERN_BEAR_ENGULF"],
            ["MOMENTUM", "BREAKOUT_UP"],
            ["MOMENTUM", "BREAKOUT_DOWN"],
        ]
        present = {k for k in evidence.keys() if (evidence.get(k, 0.0) > 0)}
        ok = False
        for combo in combos:
            if set(map(str.upper, combo)).issubset(present):
                ok = True
                break
        if not ok:
            label = "high"
    return AlertPriority(label=label, score=round(float(score), 2), evidence=evidence)
