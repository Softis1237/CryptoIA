from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..utils.horizons import horizon_to_minutes, minutes_to_horizon


@dataclass
class Signal:
    name: str
    proba_up: float  # in [0,1]
    rr: float        # expected risk:reward
    atr: float
    price: float


@dataclass
class AdaptiveRiskProfile:
    horizon_minutes: int
    horizon_label: str
    sl_atr_multiplier: float
    rr_target: float
    leverage_factor: float
    risk_multiplier: float
    valid_minutes: int
    hold_minutes: int
    notes: List[str]

    def sl_tp_distances(self, atr: float) -> tuple[float, float]:
        sl_dist = float(self.sl_atr_multiplier * atr)
        tp_dist = float(self.rr_target * sl_dist)
        return sl_dist, tp_dist


@dataclass(frozen=True)
class _DurationSummary:
    count: int
    median: Optional[float]
    p75: Optional[float]

    @classmethod
    def from_samples(cls, samples: Optional[Sequence[float]]) -> "_DurationSummary":
        values: List[float] = []
        if samples:
            for item in samples:
                if item is None:
                    continue
                try:
                    val = float(item)
                except (TypeError, ValueError):  # noqa: PERF203 - clarity outweighs
                    continue
                if val > 0:
                    values.append(val)
        if not values:
            return cls(count=0, median=None, p75=None)
        values.sort()
        median = _percentile(values, 0.5)
        p75 = _percentile(values, 0.75)
        return cls(count=len(values), median=median, p75=p75)


@dataclass(frozen=True)
class _RiskBucket:
    name: str
    max_minutes: int
    sl_atr: float
    rr_target: float
    leverage_range: Tuple[float, float]
    risk_range: Tuple[float, float]


_RISK_BUCKETS: List[_RiskBucket] = [
    _RiskBucket(
        name="ultra_short",
        max_minutes=60,
        sl_atr=0.85,
        rr_target=1.6,
        leverage_range=(0.9, 1.15),
        risk_range=(1.0, 1.3),
    ),
    _RiskBucket(
        name="short",
        max_minutes=180,
        sl_atr=1.2,
        rr_target=2.0,
        leverage_range=(0.7, 0.95),
        risk_range=(0.9, 1.15),
    ),
    _RiskBucket(
        name="swing",
        max_minutes=360,
        sl_atr=1.55,
        rr_target=2.3,
        leverage_range=(0.5, 0.75),
        risk_range=(0.8, 1.05),
    ),
    _RiskBucket(
        name="positional",
        max_minutes=1440,
        sl_atr=2.05,
        rr_target=2.7,
        leverage_range=(0.35, 0.6),
        risk_range=(0.65, 0.95),
    ),
    _RiskBucket(
        name="extended",
        max_minutes=10_080,
        sl_atr=2.4,
        rr_target=3.0,
        leverage_range=(0.25, 0.5),
        risk_range=(0.6, 0.85),
    ),
]


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


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile on empty data")
    if not 0.0 <= q <= 1.0:
        raise ValueError("Percentile q must be in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(sorted_values[int(pos)])
    lower_val = sorted_values[lower]
    upper_val = sorted_values[upper]
    weight = pos - lower
    return float(lower_val + (upper_val - lower_val) * weight)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, value)))


def _normalize_horizon(value: Union[str, int, float]) -> Tuple[int, str]:
    if isinstance(value, (int, float)):
        minutes = max(1, int(round(float(value))))
        return minutes, minutes_to_horizon(minutes)
    raw = str(value).strip().lower()
    try:
        minutes = horizon_to_minutes(raw)
        return minutes, minutes_to_horizon(minutes)
    except ValueError:
        normalized = (
            raw.replace(" ", "")
            .replace("часов", "h")
            .replace("часа", "h")
            .replace("час", "h")
            .replace("ч", "h")
            .replace("minutes", "m")
            .replace("minute", "m")
            .replace("минут", "m")
            .replace("мин", "m")
            .replace("м", "m")
        )
        minutes = horizon_to_minutes(normalized)
        return minutes, minutes_to_horizon(minutes)


def _choose_bucket(hold_minutes: int) -> _RiskBucket:
    for bucket in _RISK_BUCKETS:
        if hold_minutes <= bucket.max_minutes:
            return bucket
    return _RISK_BUCKETS[-1]


def _interpolate(range_tuple: Tuple[float, float], weight: float) -> float:
    lo, hi = range_tuple
    weight = _clip(weight, 0.0, 1.0)
    return float(lo + (hi - lo) * weight)


def _infer_hold_minutes(
    horizon_minutes: int,
    target_summary: _DurationSummary,
    stop_summary: _DurationSummary,
    confidence: float,
    min_hold_minutes: int,
    horizon_buffer_ratio: float,
) -> Tuple[int, List[str]]:
    notes: List[str] = []
    horizon_minutes = max(1, int(horizon_minutes))
    min_hold = max(1, int(min_hold_minutes))
    max_hold = max(min_hold, int(round(horizon_minutes * max(1.0, horizon_buffer_ratio))))
    if target_summary.count:
        base = target_summary.p75 or target_summary.median or float(horizon_minutes)
        notes.append(
            f"hold_from_target:{base:.1f}m_p75_samples={target_summary.count}"
        )
    elif stop_summary.count:
        base = (stop_summary.median or float(horizon_minutes)) * 0.9
        notes.append(
            f"hold_from_stop:{base:.1f}m_median_samples={stop_summary.count}"
        )
    else:
        ratio = 0.65 if horizon_minutes <= 90 else 0.85
        adaptive_ratio = ratio + (1.0 - confidence) * 0.15
        base = float(horizon_minutes) * adaptive_ratio
        notes.append(
            f"hold_fallback_ratio:{adaptive_ratio:.2f}"
        )
    hold = int(round(_clip(base, min_hold, max_hold)))
    if stop_summary.count and stop_summary.median:
        # Avoid extending well beyond historical stop-outs
        cap = int(round(stop_summary.median * 1.3))
        if cap >= min_hold:
            hold = min(hold, cap)
            notes.append(f"hold_capped_by_stop:{cap}m")
    return max(min_hold, hold), notes


def build_adaptive_risk_profile(
    signal: Signal,
    forecast_horizon: Union[str, int, float],
    *,
    leverage_cap: float = 25.0,
    historical_target_minutes: Optional[Sequence[float]] = None,
    historical_stop_minutes: Optional[Sequence[float]] = None,
    min_hold_minutes: int = 20,
    min_valid_minutes: int = 15,
    horizon_buffer_ratio: float = 1.3,
) -> AdaptiveRiskProfile:
    """Construct an adaptive risk profile driven by forecast horizon and history.

    Parameters
    ----------
    signal:
        Core predictive signal (probability, RR, ATR, price).
    forecast_horizon:
        Original forecast horizon (e.g. "4h", "30m" or minutes as an int).
    leverage_cap:
        Maximum leverage allowed by venue/config; profile outputs a factor within [0, 1.25].
    historical_target_minutes:
        Iterable durations (in minutes) describing how long similar forecasts took to reach TP.
    historical_stop_minutes:
        Iterable durations (in minutes) describing how long similar forecasts took to hit SL.
    min_hold_minutes:
        Guard-rail for minimal hold duration to prevent unrealistic micro-scales.
    min_valid_minutes:
        Minimum validity window for the recommendation.
    horizon_buffer_ratio:
        Allow holding positions beyond the forecast horizon when targets historically take longer.
    """

    forecast_minutes, forecast_label = _normalize_horizon(forecast_horizon)
    confidence = _clip(abs(signal.proba_up - 0.5) * 2.0, 0.0, 1.0)
    target_summary = _DurationSummary.from_samples(historical_target_minutes)
    stop_summary = _DurationSummary.from_samples(historical_stop_minutes)
    hold_minutes, hold_notes = _infer_hold_minutes(
        horizon_minutes=forecast_minutes,
        target_summary=target_summary,
        stop_summary=stop_summary,
        confidence=confidence,
        min_hold_minutes=min_hold_minutes,
        horizon_buffer_ratio=horizon_buffer_ratio,
    )
    bucket = _choose_bucket(hold_minutes)

    sl_atr_multiplier = bucket.sl_atr
    duration_ratio = hold_minutes / max(1.0, float(forecast_minutes))
    if duration_ratio < 0.8:
        sl_atr_multiplier *= _clip(0.85 + 0.25 * duration_ratio, 0.6, 1.05)
    elif duration_ratio > 1.1:
        sl_atr_multiplier *= _clip(1.0 + min(0.4, (duration_ratio - 1.0) * 0.35), 1.0, 1.5)

    atr_pct = signal.atr / max(signal.price, 1e-6)
    if atr_pct > 0.015:
        sl_atr_multiplier *= _clip(1.0 + (atr_pct - 0.015) * 6.0, 1.0, 1.4)
    elif atr_pct < 0.005:
        sl_atr_multiplier *= _clip(0.8 + atr_pct * 30.0, 0.6, 1.0)
    sl_atr_multiplier = _clip(sl_atr_multiplier, 0.6, 3.2)

    rr_target = bucket.rr_target
    if signal.rr > 0:
        rr_target = max(rr_target, float(signal.rr))
    rr_target *= _clip(0.9 + confidence * 0.2, 0.9, 1.2)
    rr_target = _clip(rr_target, 1.2, 4.5)

    leverage_factor = _interpolate(bucket.leverage_range, confidence)
    if forecast_minutes <= 90:
        leverage_factor *= 1.1
    elif forecast_minutes >= 720:
        leverage_factor *= 0.88
    if atr_pct > 0.012:
        leverage_factor *= _clip(1.0 - (atr_pct - 0.012) * 8.0, 0.4, 1.0)
    leverage_factor = _clip(leverage_factor, 0.2, 1.2)

    risk_multiplier = _interpolate(bucket.risk_range, confidence)
    if forecast_minutes >= 720:
        risk_multiplier *= 0.9
    if atr_pct > 0.02:
        risk_multiplier *= _clip(1.0 - (atr_pct - 0.02) * 5.0, 0.5, 1.0)
    risk_multiplier = _clip(risk_multiplier, 0.55, 1.35)

    valid_ratio = 0.5 + confidence * 0.25
    valid_minutes = int(round(_clip(hold_minutes * valid_ratio, min_valid_minutes, hold_minutes)))

    notes: List[str] = []
    notes.append(f"bucket:{bucket.name}")
    notes.append(f"confidence:{confidence:.2f}")
    notes.append(f"atr_pct:{atr_pct:.4f}")
    notes.append(f"leverage_cap:{leverage_cap}")
    notes.append(f"forecast_horizon:{forecast_label}")
    notes.extend(hold_notes)

    return AdaptiveRiskProfile(
        horizon_minutes=hold_minutes,
        horizon_label=minutes_to_horizon(hold_minutes),
        sl_atr_multiplier=sl_atr_multiplier,
        rr_target=rr_target,
        leverage_factor=_clip(leverage_factor, 0.2, 1.2),
        risk_multiplier=risk_multiplier,
        valid_minutes=valid_minutes,
        hold_minutes=hold_minutes,
        notes=notes,
    )
