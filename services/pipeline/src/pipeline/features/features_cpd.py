from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def enrich_with_cpd_simple(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Lightweight CPD fallback without external deps.

    Detects level shifts on close via rolling z-score of mean differences.
    Controlled by USE_CPD_SIMPLE=1 and CPD_Z_THRESHOLD (default 2.5).
    Adds columns: cp_simple_flag (0/1), cpd_score (|z|).
    """
    if os.getenv("USE_CPD_SIMPLE", "0") not in {"1", "true", "True"}:
        return df, {"cpd_enabled": False}
    try:
        work = df.copy()
        close = work["close"].astype(float)
        win = max(10, int(os.getenv("CPD_WINDOW", "30")))
        if len(close) < 2 * win + 5:
            work["cp_simple_flag"] = 0
            work["cpd_score"] = 0.0
            return work, {"cpd_enabled": True, "msg": "insufficient length"}
        m1 = close.rolling(win).mean()
        m2 = close.shift(win).rolling(win).mean()
        diff = (m2 - m1)
        sd = close.rolling(win).std(ddof=0)
        z = diff / (sd + 1e-9)
        z = z.fillna(0.0)
        thr = float(os.getenv("CPD_Z_THRESHOLD", "2.5"))
        flag = (z.abs() >= thr).astype(int)
        work["cpd_score"] = z.abs().astype(float)
        work["cp_simple_flag"] = flag
        return work, {"cpd_enabled": True, "threshold": thr}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"CPD simple failed: {e}")
        return df, {"cpd_enabled": False, "error": str(e)}

