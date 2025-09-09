from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Signal:
    name: str
    proba_up: float  # in [0,1]
    rr: float        # expected risk:reward
    atr: float
    price: float


def kelly_fraction(p: float, b: float) -> float:
    """Kelly fraction for a simple Bernoulli with odds b (TP/SL ratio)."""
    q = 1.0 - p
    f = (p * (b + 1) - 1) / b if b > 0 else 0.0
    return max(0.0, min(1.0, f))


def position_size_equal_risk(equity: float, risk_per_trade: float, price: float, sl_price: float) -> float:
    risk_amount = equity * risk_per_trade
    return float(risk_amount / max(1e-9, abs(price - sl_price)))


def allocate_across_signals(signals: List[Signal], equity: float, risk_per_trade: float, max_positions: int = 3, scheme: str = "kelly") -> Dict[str, float]:
    """Return allocation weights per signal name (fraction of equity)."""
    top = sorted(signals, key=lambda s: s.proba_up * s.rr, reverse=True)[:max_positions]
    if not top:
        return {}
    if scheme == "equal":
        w = 1.0 / len(top)
        return {s.name: w for s in top}
    # kelly-based on odds rr (approx)
    scores = [kelly_fraction(s.proba_up, s.rr) for s in top]
    total = sum(scores) or 1.0
    return {s.name: v / total for s, v in zip(top, scores)}

