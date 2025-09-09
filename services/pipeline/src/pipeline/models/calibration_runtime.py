from __future__ import annotations

import os
import io
import pickle
from functools import lru_cache
from typing import Optional

from ..infra.s3 import download_bytes


@lru_cache(maxsize=2)
def _load_isotonic(horizon: str) -> Optional[object]:
    key = os.getenv("ISOTONIC_S3_4H") if horizon == "4h" else os.getenv("ISOTONIC_S3_12H")
    if not key:
        return None
    try:
        raw = download_bytes(key)
        return pickle.loads(raw)
    except Exception:
        return None


def calibrate_proba(prob: float, horizon: str) -> float:
    mdl = _load_isotonic(horizon)
    if mdl is None:
        return float(prob)
    try:
        import numpy as np
        p = float(mdl.predict([prob])[0])
        return max(0.0, min(1.0, p))
    except Exception:
        return float(prob)

