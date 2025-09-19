from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .pivot_analysis import find_pivots


@dataclass
class PatternResult:
    name: str
    status: str  # "in_progress" | "confirmed" | "failed"
    direction: Optional[str]
    neckline: Optional[float]
    breakout_price: Optional[float]
    left_shoulder: Optional[int]
    head: Optional[int]
    right_shoulder: Optional[int]
    meta: Dict[str, Any]


def _empty_result(name: str) -> PatternResult:
    return PatternResult(
        name=name,
        status="failed",
        direction=None,
        neckline=None,
        breakout_price=None,
        left_shoulder=None,
        head=None,
        right_shoulder=None,
        meta={},
    )


def _prepare_pivots(pivots: pd.DataFrame | None) -> pd.DataFrame:
    if pivots is None or pivots.empty:
        return pd.DataFrame(columns=["idx", "price", "kind"])
    required = {"idx", "price", "kind"}
    missing = required.difference(pivots.columns)
    if missing:
        raise ValueError(f"pivot DataFrame must contain columns {required}, missing {missing}")
    res = pivots.copy()
    res["idx"] = res["idx"].astype(int)
    res["price"] = res["price"].astype(float)
    res["kind"] = res["kind"].astype(str)
    return res.sort_values("idx").reset_index(drop=True)


def _ensure_price_series(prices: pd.Series | None) -> pd.Series | None:
    if prices is None or prices.empty:
        return None
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    try:
        return prices.astype(float)
    except Exception:
        return prices


def _build_hs_candidate(
    window: pd.DataFrame,
    *,
    variant: str,
    shoulder_tol: float,
    head_margin: float,
    min_separation: int,
) -> Dict[str, Any] | None:
    idx_seq = window["idx"].astype(int).tolist()
    if any(idx_seq[i + 1] <= idx_seq[i] for i in range(len(idx_seq) - 1)):
        return None
    if any((idx_seq[i + 1] - idx_seq[i]) < min_separation for i in range(len(idx_seq) - 1)):
        return None

    records = window.to_dict(orient="records")

    if variant == "classic":
        h1, l1, h2, l2, h3 = records
        shoulder_prices = (float(h1["price"]), float(h3["price"]))
        avg_shoulder = sum(shoulder_prices) / 2.0 or max(shoulder_prices)
        if avg_shoulder <= 0:
            avg_shoulder = max(1.0, shoulder_prices[0], shoulder_prices[1])
        symmetry = abs(shoulder_prices[0] - shoulder_prices[1]) / max(1e-9, avg_shoulder)
        if symmetry > shoulder_tol:
            return None
        head_price = float(h2["price"])
        max_shoulder = max(shoulder_prices)
        head_margin_ratio = (head_price - max_shoulder) / max(1e-9, max_shoulder)
        if head_margin_ratio < head_margin:
            return None
        neckline = (float(l1["price"]) + float(l2["price"])) / 2.0
        if neckline >= max_shoulder:
            return None
        direction = "bearish"
        pattern_name = "head_and_shoulders"
        neckline_points = [float(l1["price"]), float(l2["price"])]
    elif variant == "inverse":
        l1, h1, l2, h2, l3 = records
        shoulder_prices = (float(l1["price"]), float(l3["price"]))
        avg_shoulder = sum(shoulder_prices) / 2.0 or max(abs(shoulder_prices[0]), abs(shoulder_prices[1]), 1e-9)
        symmetry = abs(shoulder_prices[0] - shoulder_prices[1]) / max(1e-9, avg_shoulder)
        if symmetry > shoulder_tol:
            return None
        head_price = float(l2["price"])
        min_shoulder = min(shoulder_prices)
        denom = min_shoulder if min_shoulder != 0 else max(1e-9, avg_shoulder)
        head_margin_ratio = (min_shoulder - head_price) / max(1e-9, denom)
        if head_margin_ratio < head_margin:
            return None
        neckline = (float(h1["price"]) + float(h2["price"])) / 2.0
        if neckline <= min(shoulder_prices):
            return None
        direction = "bullish"
        pattern_name = "inverse_head_and_shoulders"
        neckline_points = [float(h1["price"]), float(h2["price"])]
    else:  # pragma: no cover - defensive branch
        return None

    return {
        "pattern_name": pattern_name,
        "direction": direction,
        "neckline": float(neckline),
        "left_idx": idx_seq[0],
        "head_idx": idx_seq[2],
        "right_idx": idx_seq[4],
        "last_idx": idx_seq[-1],
        "meta": {
            "variant": variant,
            "pivot_indices": idx_seq,
            "sequence": records,
            "shoulder_prices": shoulder_prices,
            "shoulder_symmetry": 1.0 - symmetry,
            "head_margin_ratio": head_margin_ratio,
            "neckline_points": neckline_points,
        },
    }


def _detect_hs_variant_from_pivots(
    pivots: pd.DataFrame,
    prices: pd.Series | None,
    *,
    variant: str,
    shoulder_tol: float,
    head_margin: float,
    min_separation: int,
    neckline_tol: float,
) -> PatternResult:
    base_name = "head_and_shoulders" if variant == "classic" else "inverse_head_and_shoulders"
    piv = _prepare_pivots(pivots)
    if piv.empty or len(piv) < 5:
        return _empty_result(base_name)

    seq_target = ["high", "low", "high", "low", "high"] if variant == "classic" else ["low", "high", "low", "high", "low"]
    candidates: list[Dict[str, Any]] = []
    for start in range(len(piv) - 4):
        window = piv.iloc[start : start + 5]
        if window["kind"].tolist() != seq_target:
            continue
        candidate = _build_hs_candidate(
            window,
            variant=variant,
            shoulder_tol=shoulder_tol,
            head_margin=head_margin,
            min_separation=min_separation,
        )
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return _empty_result(base_name)

    chosen = max(candidates, key=lambda c: (c["last_idx"], c["meta"].get("head_margin_ratio", 0.0)))

    prices = _ensure_price_series(prices)
    last_price = None
    if prices is not None:
        value = prices.iloc[-1]
        if not pd.isna(value):
            last_price = float(value)

    status = "in_progress"
    breakout_price = None
    neckline = float(chosen["neckline"])
    if last_price is not None:
        if variant == "classic":
            confirmed = last_price <= neckline * (1 - neckline_tol)
        else:
            confirmed = last_price >= neckline * (1 + neckline_tol)
        if confirmed:
            status = "confirmed"
            breakout_price = last_price

    meta = {**chosen["meta"], "candidates": len(candidates)}
    if last_price is not None:
        meta["last_price"] = last_price

    return PatternResult(
        name=base_name,
        status=status,
        direction="bearish" if variant == "classic" else "bullish",
        neckline=neckline,
        breakout_price=breakout_price,
        left_shoulder=chosen["left_idx"],
        head=chosen["head_idx"],
        right_shoulder=chosen["right_idx"],
        meta=meta,
    )


def detect_head_and_shoulders_from_pivots(
    pivots: pd.DataFrame,
    prices: pd.Series | None = None,
    *,
    shoulder_tol: float = 0.02,
    head_margin: float = 0.01,
    min_separation: int = 2,
    neckline_tol: float = 0.01,
) -> PatternResult:
    """Detect a classic Head & Shoulders pattern given pivot points."""

    return _detect_hs_variant_from_pivots(
        pivots,
        prices,
        variant="classic",
        shoulder_tol=shoulder_tol,
        head_margin=head_margin,
        min_separation=min_separation,
        neckline_tol=neckline_tol,
    )


def detect_inverse_head_and_shoulders_from_pivots(
    pivots: pd.DataFrame,
    prices: pd.Series | None = None,
    *,
    shoulder_tol: float = 0.02,
    head_margin: float = 0.01,
    min_separation: int = 2,
    neckline_tol: float = 0.01,
) -> PatternResult:
    """Detect an Inverse Head & Shoulders pattern given pivot points."""

    return _detect_hs_variant_from_pivots(
        pivots,
        prices,
        variant="inverse",
        shoulder_tol=shoulder_tol,
        head_margin=head_margin,
        min_separation=min_separation,
        neckline_tol=neckline_tol,
    )


def detect_head_and_shoulders(
    ohlcv: pd.DataFrame | None,
    left_bars: int = 3,
    right_bars: int = 3,
    shoulder_tol: float = 0.02,
    neckline_tol: float = 0.01,
    *,
    pivots: pd.DataFrame | None = None,
    head_margin: float = 0.01,
    min_separation: int = 2,
) -> PatternResult:
    """Wrapper that computes pivots from OHLCV before detecting classic H&S."""

    if pivots is None:
        if ohlcv is None:
            raise ValueError("Either ohlcv or pivots must be provided for detection.")
        pivots = find_pivots(ohlcv, left_bars=left_bars, right_bars=right_bars)
    prices = ohlcv["close"] if (ohlcv is not None and "close" in ohlcv.columns) else None
    return detect_head_and_shoulders_from_pivots(
        pivots,
        prices,
        shoulder_tol=shoulder_tol,
        head_margin=head_margin,
        min_separation=min_separation,
        neckline_tol=neckline_tol,
    )


def detect_inverse_head_and_shoulders(
    ohlcv: pd.DataFrame | None,
    left_bars: int = 3,
    right_bars: int = 3,
    shoulder_tol: float = 0.02,
    neckline_tol: float = 0.01,
    *,
    pivots: pd.DataFrame | None = None,
    head_margin: float = 0.01,
    min_separation: int = 2,
) -> PatternResult:
    """Wrapper that computes pivots from OHLCV before detecting inverse H&S."""

    if pivots is None:
        if ohlcv is None:
            raise ValueError("Either ohlcv or pivots must be provided for detection.")
        pivots = find_pivots(ohlcv, left_bars=left_bars, right_bars=right_bars)
    prices = ohlcv["close"] if (ohlcv is not None and "close" in ohlcv.columns) else None
    return detect_inverse_head_and_shoulders_from_pivots(
        pivots,
        prices,
        shoulder_tol=shoulder_tol,
        head_margin=head_margin,
        min_separation=min_separation,
        neckline_tol=neckline_tol,
    )


def detect_double_top_bottom(
    ohlcv: pd.DataFrame | None,
    left_bars: int = 3,
    right_bars: int = 3,
    height_tol: float = 0.01,
    depth_min: float = 0.005,
    *,
    pivots: pd.DataFrame | None = None,
    close_series: pd.Series | None = None,
) -> PatternResult:
    piv = pivots
    if piv is None:
        if ohlcv is None:
            return _empty_result("double_top_bottom")
        piv = find_pivots(ohlcv, left_bars=left_bars, right_bars=right_bars)
    piv = _prepare_pivots(piv)
    if piv.empty:
        return _empty_result("double_top_bottom")

    highs = piv[piv["kind"] == "high"].sort_values("idx")
    lows = piv[piv["kind"] == "low"].sort_values("idx")

    close = close_series
    if close is None and ohlcv is not None and "close" in ohlcv.columns:
        close = ohlcv["close"]
    close = _ensure_price_series(close)
    close_val = None
    if close is not None:
        value = close.iloc[-1]
        if not pd.isna(value):
            close_val = float(value)

    # Double top detection
    if len(highs) >= 2:
        a = highs.iloc[-2]
        b = highs.iloc[-1]
        avg = (float(a["price"]) + float(b["price"])) / 2.0
        if avg > 0 and abs(float(a["price"]) - float(b["price"])) / max(1e-9, avg) <= height_tol:
            mid = lows[(lows["idx"] > a["idx"]) & (lows["idx"] < b["idx"])].sort_values("idx")
            if len(mid) >= 1:
                neckline = float(mid.iloc[-1]["price"])
                depth = (min(float(a["price"]), float(b["price"])) - neckline) / max(1e-9, avg)
                if depth >= depth_min:
                    status = "in_progress"
                    breakout_price = None
                    if close_val is not None and close_val < neckline:
                        status = "confirmed"
                        breakout_price = close_val
                    return PatternResult(
                        name="double_top",
                        status=status,
                        direction="bearish",
                        neckline=neckline,
                        breakout_price=breakout_price,
                        left_shoulder=int(a["idx"]),
                        head=None,
                        right_shoulder=int(b["idx"]),
                        meta={
                            "height_tol": height_tol,
                            "depth": depth,
                            "pivot_indices": piv.to_dict(orient="records"),
                        },
                    )

    # Double bottom detection
    if len(lows) >= 2:
        a = lows.iloc[-2]
        b = lows.iloc[-1]
        avg = (float(a["price"]) + float(b["price"])) / 2.0
        if avg != 0 and abs(float(a["price"]) - float(b["price"])) / max(1e-9, abs(avg)) <= height_tol:
            mid = highs[(highs["idx"] > a["idx"]) & (highs["idx"] < b["idx"])].sort_values("idx")
            if len(mid) >= 1:
                neckline = float(mid.iloc[-1]["price"])
                depth = (neckline - max(float(a["price"]), float(b["price"]))) / max(1e-9, abs(avg))
                if depth >= depth_min:
                    status = "in_progress"
                    breakout_price = None
                    if close_val is not None and close_val > neckline:
                        status = "confirmed"
                        breakout_price = close_val
                    return PatternResult(
                        name="double_bottom",
                        status=status,
                        direction="bullish",
                        neckline=neckline,
                        breakout_price=breakout_price,
                        left_shoulder=int(a["idx"]),
                        head=None,
                        right_shoulder=int(b["idx"]),
                        meta={
                            "height_tol": height_tol,
                            "depth": depth,
                            "pivot_indices": piv.to_dict(orient="records"),
                        },
                    )

    return _empty_result("double_top_bottom")


def detect_triangle(
    ohlcv: pd.DataFrame | None,
    left_bars: int = 3,
    right_bars: int = 3,
    window: int = 80,
    narrowing_ratio: float = 0.6,
    breakout_tol: float = 0.001,
) -> PatternResult:
    if ohlcv is None or ohlcv.empty or len(ohlcv) < window:
        return _empty_result("triangle")

    import numpy as np

    sub = ohlcv.tail(window).reset_index(drop=True)
    piv = find_pivots(sub, left_bars=left_bars, right_bars=right_bars)
    piv = _prepare_pivots(piv)
    if piv.empty:
        return _empty_result("triangle")

    highs_idx = piv[piv["kind"] == "high"]["idx"].astype(int).tolist()
    lows_idx = piv[piv["kind"] == "low"]["idx"].astype(int).tolist()
    if len(highs_idx) < 2 or len(lows_idx) < 2:
        return _empty_result("triangle")

    x = np.arange(len(sub))
    hi_vals = sub["high"].astype(float).values
    lo_vals = sub["low"].astype(float).values

    hi_idx = np.array(highs_idx[-3:])
    lo_idx = np.array(lows_idx[-3:])
    if len(hi_idx) < 2 or len(lo_idx) < 2:
        return _empty_result("triangle")

    A_hi = np.vstack([hi_idx, np.ones(len(hi_idx))]).T
    m_hi, b_hi = np.linalg.lstsq(A_hi, hi_vals[hi_idx], rcond=None)[0]
    A_lo = np.vstack([lo_idx, np.ones(len(lo_idx))]).T
    m_lo, b_lo = np.linalg.lstsq(A_lo, lo_vals[lo_idx], rcond=None)[0]

    width_start = (m_hi * 0 + b_hi) - (m_lo * 0 + b_lo)
    width_end = (m_hi * (len(sub) - 1) + b_hi) - (m_lo * (len(sub) - 1) + b_lo)
    if width_start <= 0 or width_end <= 0:
        return _empty_result("triangle")
    if (width_end / max(1e-9, width_start)) >= narrowing_ratio:
        return _empty_result("triangle")

    close = float(sub["close"].astype(float).iloc[-1])
    hi_last = float(m_hi * (len(sub) - 1) + b_hi)
    lo_last = float(m_lo * (len(sub) - 1) + b_lo)

    if close > hi_last * (1.0 + breakout_tol):
        return PatternResult(
            name="triangle",
            status="confirmed",
            direction="bullish",
            neckline=None,
            breakout_price=close,
            left_shoulder=None,
            head=None,
            right_shoulder=None,
            meta={"slope_hi": m_hi, "slope_lo": m_lo, "width_ratio": width_end / max(1e-9, width_start)},
        )
    if close < lo_last * (1.0 - breakout_tol):
        return PatternResult(
            name="triangle",
            status="confirmed",
            direction="bearish",
            neckline=None,
            breakout_price=close,
            left_shoulder=None,
            head=None,
            right_shoulder=None,
            meta={"slope_hi": m_hi, "slope_lo": m_lo, "width_ratio": width_end / max(1e-9, width_start)},
        )

    return PatternResult(
        name="triangle",
        status="in_progress",
        direction=None,
        neckline=None,
        breakout_price=None,
        left_shoulder=None,
        head=None,
        right_shoulder=None,
        meta={"slope_hi": m_hi, "slope_lo": m_lo, "width_ratio": width_end / max(1e-9, width_start)},
    )

