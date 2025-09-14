from __future__ import annotations

from typing import Dict, List
import os

import pandas as pd

from ..infra.db import fetch_technical_patterns


def _safe_abs(x: pd.Series) -> pd.Series:
    try:
        return (x.astype(float)).abs()
    except Exception:
        return x.abs()


def detect_patterns(df: pd.DataFrame, patterns: List[dict] | None = None) -> Dict[str, pd.Series]:
    """Detect a minimal set of classic patterns over the whole series.

    Returns mapping column_name -> 0/1 pd.Series aligned to df index.
    """
    if df.empty:
        return {}
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    body = _safe_abs(c - o)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    eps = 1e-9

    # Hammer / Shooting star
    hammer = ((lower >= 2.0 * body) & (upper <= 0.4 * body)).astype(int)
    shooting = ((upper >= 2.0 * body) & (lower <= 0.4 * body)).astype(int)

    # Engulfing
    o1, c1 = o.shift(1), c.shift(1)
    body1 = _safe_abs(c1 - o1)
    green = (c > o)
    red = (c < o)
    prev_green = (c1 > o1)
    prev_red = (c1 < o1)
    bull_engulf = (
        prev_red & green & (o <= c1) & (c >= o1) & (_safe_abs(c - o) > 1.05 * body1)
    ).astype(int)
    bear_engulf = (
        prev_green & red & (o >= c1) & (c <= o1) & (_safe_abs(c - o) > 1.05 * body1)
    ).astype(int)

    # Morning Star (approximate heuristic)
    o2, c2 = o.shift(2), c.shift(2)
    body2 = _safe_abs(c2 - o2)
    small_mid = _safe_abs(c1 - o1) < (body2 * 0.6)
    c1_red = c2 < o2
    c3_green = green
    close3_above_mid1 = c > ((o2 + c2) / 2.0)
    morning_star = (c1_red & small_mid & c3_green & close3_above_mid1).astype(int)

    # Doji
    range_total = h - l + eps
    doji = (body <= (range_total * 0.1)).astype(int)

    # Harami (bullish / bearish)
    harami_bull = (
        prev_red
        & green
        & (o >= c1)
        & (c <= o1)
        & (body < body1)
    ).astype(int)
    harami_bear = (
        prev_green
        & red
        & (o <= c1)
        & (c >= o1)
        & (body < body1)
    ).astype(int)

    # Three White Soldiers
    green2 = c2 > o2
    open1_in2 = (o1 > o2) & (o1 < c2)
    open_in1 = (o > o1) & (o < c1)
    close_progress = (c2 < c1) & (c1 < c)
    tws = (green2 & prev_green & green & open1_in2 & open_in1 & close_progress).astype(int)

    # Aggregate score (candlestick)
    score = (
        hammer
        + shooting
        + bull_engulf
        + bear_engulf
        + morning_star
        + doji
        + harami_bull
        + harami_bear
        + tws
    )

    out = {
        "pat_hammer": hammer,
        "pat_shooting_star": shooting,
        "pat_engulfing_bull": bull_engulf,
        "pat_engulfing_bear": bear_engulf,
        "pat_morning_star": morning_star,
        "pat_doji": doji,
        "pat_harami_bull": harami_bull,
        "pat_harami_bear": harami_bear,
        "pat_three_white_soldiers": tws,
    }

    # Chart patterns (evaluate over recent window; set flag at last index)
    try:
        win = int((patterns[0].get("definition", {}) or {}).get("lookback", 240)) if patterns else 240
    except Exception:
        win = 240
    W = min(len(df), max(60, win))
    if W >= 60:
        cw = c.tail(W).reset_index(drop=True)
        # Optional volume confirmation via z-score if present
        vol_z_last = None
        try:
            if "volume_z" in df.columns:
                vol_z_last = float(df["volume_z"].astype(float).iloc[-1])
        except Exception:
            vol_z_last = None
        # local extrema indices
        import numpy as np
        def _loc_max(x: pd.Series) -> List[int]:
            x = x.values
            idx = []
            for i in range(1, len(x) - 1):
                if x[i] > x[i - 1] and x[i] > x[i + 1]:
                    idx.append(i)
            return idx
        def _loc_min(x: pd.Series) -> List[int]:
            x = x.values
            idx = []
            for i in range(1, len(x) - 1):
                if x[i] < x[i - 1] and x[i] < x[i + 1]:
                    idx.append(i)
            return idx
        pidx = _loc_max(cw)
        midx = _loc_min(cw)
        # thresholds can be tuned via ENV (values in fractions / bars)
        try:
            tol = float(os.getenv("PAT_TOL_BPS", "50")) / 1e4
        except Exception:
            tol = 0.005
        try:
            depth = float(os.getenv("PAT_DEPTH_BPS", "30")) / 1e4
        except Exception:
            depth = 0.003
        try:
            min_sep = int(float(os.getenv("PAT_MIN_SEP", "10")))
        except Exception:
            min_sep = 10

        dbl_top = 0
        if len(pidx) >= 2:
            # pick top-2 peaks by height with separation
            peaks = sorted([(i, float(cw[i])) for i in pidx], key=lambda t: t[1], reverse=True)
            # try first 3 candidates to find two separated peaks of similar height
            for i1 in range(min(3, len(peaks))):
                for i2 in range(i1 + 1, min(4, len(peaks))):
                    a, pa = peaks[i1]
                    b, pb = peaks[i2]
                    if abs(a - b) < min_sep:
                        continue
                    if abs(pa - pb) / max(1e-9, (pa + pb) / 2.0) <= tol:
                        lo, hi = sorted([a, b])
                        trough = min((float(cw[j]) for j in midx if lo < j < hi), default=float(cw[min(lo, hi)]))
                        neckline = trough
                        # confirmation: last close below neckline
                        confirmed = float(cw.iloc[-1]) < (neckline * (1.0 - 0.001))
                        if vol_z_last is not None:
                            confirmed = confirmed and (vol_z_last > 0)
                        if ((min(pa, pb) - neckline) / max(1e-9, min(pa, pb)) >= depth) and confirmed:
                            dbl_top = 1
                            break
                if dbl_top:
                    break

        dbl_bot = 0
        if len(midx) >= 2:
            troughs = sorted([(i, float(cw[i])) for i in midx], key=lambda t: t[1])  # lowest first
            for i1 in range(min(3, len(troughs))):
                for i2 in range(i1 + 1, min(4, len(troughs))):
                    a, pa = troughs[i1]
                    b, pb = troughs[i2]
                    if abs(a - b) < min_sep:
                        continue
                    if abs(pa - pb) / max(1e-9, (pa + pb) / 2.0) <= tol:
                        lo, hi = sorted([a, b])
                        peak = max((float(cw[j]) for j in pidx if lo < j < hi), default=float(cw[max(lo, hi)]))
                        neckline = peak
                        # confirmation: last close above neckline
                        confirmed = float(cw.iloc[-1]) > (neckline * (1.0 + 0.001))
                        if vol_z_last is not None:
                            confirmed = confirmed and (vol_z_last > 0)
                        if ((neckline - max(pa, pb)) / max(1e-9, neckline) >= depth) and confirmed:
                            dbl_bot = 1
                            break
                if dbl_bot:
                    break

        hns = 0
        if len(pidx) >= 3:
            # head = highest peak, shoulders = best peaks on sides with similar height
            peaks_sorted = sorted([(i, float(cw[i])) for i in pidx], key=lambda t: t[1], reverse=True)
            head_i, head_v = peaks_sorted[0]
            lefts = [t for t in pidx if t < head_i]
            rights = [t for t in pidx if t > head_i]
            if lefts and rights:
                # choose nearest shoulders around head respecting minimal separation
                l_candidates = [i for i in lefts if head_i - i >= min_sep]
                r_candidates = [i for i in rights if i - head_i >= min_sep]
                if l_candidates and r_candidates:
                    l_i = max(l_candidates)
                    r_i = min(r_candidates)
                else:
                    l_i = max(lefts)
                    r_i = min(rights)
                l_v = float(cw[l_i])
                r_v = float(cw[r_i])
                shoulders_close = abs(l_v - r_v) / max(1e-9, (l_v + r_v) / 2.0) <= 0.02
                head_higher = (head_v > l_v) and (head_v > r_v) and ((head_v - max(l_v, r_v)) / max(1e-9, head_v) >= 0.01)
                # neckline via two minima between shoulders/head
                seg1 = cw[min(l_i, head_i):max(l_i, head_i)+1]
                seg2 = cw[min(head_i, r_i):max(head_i, r_i)+1]
                n1 = float(seg1.min()) if len(seg1) else float('inf')
                n2 = float(seg2.min()) if len(seg2) else float('inf')
                neckline = (n1 + n2) / 2.0 if n1 < float('inf') and n2 < float('inf') else None
                confirmed = False
                if neckline is not None:
                    confirmed = float(cw.iloc[-1]) < (neckline * (1.0 - 0.001))
                    if vol_z_last is not None:
                        confirmed = confirmed and (vol_z_last > 0)
                hns = 1 if (shoulders_close and head_higher and confirmed) else 0

        # build series with flag on last index
        zeros = pd.Series(0, index=df.index)
        out["pat_double_top_recent"] = zeros.copy()
        out["pat_double_bottom_recent"] = zeros.copy()
        out["pat_head_and_shoulders_recent"] = zeros.copy()
        out["pat_double_top_recent"].iloc[-1] = dbl_top
        out["pat_double_bottom_recent"].iloc[-1] = dbl_bot
        out["pat_head_and_shoulders_recent"].iloc[-1] = hns

        # Triangles / Wedges / Flags (coarse)
        try:
            import numpy as np
            hw = df["high"].astype(float).tail(W).reset_index(drop=True)
            lw = df["low"].astype(float).tail(W).reset_index(drop=True)
            cw = df["close"].astype(float).tail(W).reset_index(drop=True)

            def _slope(x: pd.Series) -> float:
                xs = np.arange(len(x))
                A = np.vstack([xs, np.ones(len(xs))]).T
                m, _ = np.linalg.lstsq(A, x.values, rcond=None)[0]
                return float(m)

            s_hi = _slope(hw)
            s_lo = _slope(lw)
            s_full = _slope(cw)
            k = max(10, int(W / 3))
            width_start = float(hw.head(k).max() - lw.head(k).min())
            width_end = float(hw.tail(k).max() - lw.tail(k).min())
            narrowing = (width_end / max(1e-9, width_start)) < 0.6
            tri = (s_hi < 0) and (s_lo > 0) and narrowing
            wedge = ((s_hi > 0 and s_lo > 0) or (s_hi < 0 and s_lo < 0)) and narrowing

            # Flag: strong trend followed by small parallel channel
            s_hi_last = _slope(hw.tail(k))
            s_lo_last = _slope(lw.tail(k))
            parallel = (
                (s_hi_last * s_lo_last > 0)
                and (abs(s_hi_last - s_lo_last) / max(1e-9, abs(s_hi_last) + abs(s_lo_last)) < 0.2)
            )
            consolidation = (
                abs(s_hi_last) < abs(s_full) * 0.5
                and abs(s_lo_last) < abs(s_full) * 0.5
            )
            flag = (abs(s_full) > 0.01) and parallel and consolidation

            zeros2 = pd.Series(0, index=df.index)
            out["pat_triangle_recent"] = zeros2.copy()
            out["pat_wedge_recent"] = zeros2.copy()
            out["pat_flag_recent"] = zeros2.copy()
            out["pat_triangle_recent"].iloc[-1] = int(1 if tri else 0)
            out["pat_wedge_recent"].iloc[-1] = int(1 if wedge else 0)
            out["pat_flag_recent"].iloc[-1] = int(1 if flag else 0)
        except Exception:
            pass

    # Final score with all patterns
    score = (
        score
        + out.get("pat_double_top_recent", pd.Series(0, index=df.index))
        + out.get("pat_double_bottom_recent", pd.Series(0, index=df.index))
        + out.get("pat_head_and_shoulders_recent", pd.Series(0, index=df.index))
        + out.get("pat_triangle_recent", pd.Series(0, index=df.index))
        + out.get("pat_wedge_recent", pd.Series(0, index=df.index))
        + out.get("pat_flag_recent", pd.Series(0, index=df.index))
    )
    out["pat_score"] = score

    return out


def load_patterns_from_db() -> List[dict]:
    """Fetch raw pattern definitions (not yet used by rules above)."""
    try:
        return fetch_technical_patterns()
    except Exception:
        return []
