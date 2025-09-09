from __future__ import annotations

from typing import Dict, Tuple, Optional


def verify(
    card: Dict,
    current_price: float,
    leverage_cap: float = 25.0,
    interval: Optional[Tuple[float, float]] = None,
    atr: Optional[float] = None,
) -> Tuple[bool, str]:
    if not card or card.get("side") in (None, "NO-TRADE"):
        return False, "no_trade"
    side = card.get("side")
    entry_zone = card.get("entry", {}).get("zone") or [current_price, current_price]
    sl = float(card.get("stop_loss", current_price))
    tp = float(card.get("take_profit", current_price))
    lev = float(card.get("leverage", 1.0))
    conf = float(card.get("confidence", 0.0))
    if lev > leverage_cap:
        return False, "leverage_cap_exceeded"
    if conf < 0.2:
        return False, "low_confidence"
    if side == "LONG" and sl >= min(entry_zone):
        return False, "invalid_sl_long"
    if side == "SHORT" and sl <= max(entry_zone):
        return False, "invalid_sl_short"
    if abs(tp - min(entry_zone)) < abs(min(entry_zone) - sl) * 0.5:
        return False, "tp_too_close"
    # Interval width vs price / ATR checks (volatility safety)
    if interval:
        low, high = float(interval[0]), float(interval[1])
        width = max(0.0, high - low)
        if width / max(1e-6, current_price) > 0.02:  # >2% wide interval for 4h is too uncertain
            return False, "interval_too_wide"
        if atr is not None and width / max(1e-6, atr) > 10.0:
            return False, "interval_vs_atr_incoherent"
    # Approx liquidation proximity (heuristic, linear perp)
    # Use conservative buffer = 0.9 / lev (10% maintenance cushion)
    lev = max(1.0, lev)
    buffer = 0.9 / lev
    if side == "LONG":
        liq = min(entry_zone) * (1 - buffer)
        if sl <= liq * 1.005:  # SL at/below liq
            return False, "sl_near_liquidation"
    if side == "SHORT":
        liq = max(entry_zone) * (1 + buffer)
        if sl >= liq * 0.995:
            return False, "sl_near_liquidation"
    return True, "OK"
