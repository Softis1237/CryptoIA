from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..infra.s3 import download_bytes


@dataclass
class ScenarioModelerInput:
    features_path_s3: str
    current_price: float
    atr: float
    regime: str | None = None
    slot: str = "manual"


def _levels(current_price: float, atr: float) -> List[float]:
    # Simple R/S levels around price by ATR multiples
    R1 = current_price + atr
    R2 = current_price + 2 * atr
    S1 = current_price - atr
    S2 = current_price - 2 * atr
    return [round(S2, 2), round(S1, 2), round(current_price, 2), round(R1, 2), round(R2, 2)]


def run(inp: ScenarioModelerInput) -> tuple[List[dict], List[float]]:
    lvls = _levels(inp.current_price, max(1e-6, inp.atr))
    S2, S1, P, R1, R2 = lvls

    # Five scenarios with probabilities (sum≈1.0)
    scenarios: List[dict] = [
        {"if_level": f"Пробой {R1}", "then_path": f"{R1} → {R2}", "prob": 0.25, "invalidation": f"Закрытие ниже {S1}"},
        {"if_level": f"Отбой от {R1}", "then_path": f"{R1} → {P} → {S1}", "prob": 0.20, "invalidation": f"Устойчивое закрепление выше {R1}"},
        {"if_level": f"Флэт {S1}–{R1}", "then_path": f"{S1} ↔ {R1}", "prob": 0.30, "invalidation": f"Выход за {S1}/{R1}"},
        {"if_level": f"Пробой {S1}", "then_path": f"{S1} → {S2}", "prob": 0.15, "invalidation": f"Возврат выше {P}"},
        {"if_level": f"Ложный пробой {S1}", "then_path": f"{S1} → {P}", "prob": 0.10, "invalidation": f"Закрепление ниже {S1}"},
    ]
    # normalize probabilities
    total = sum(s["prob"] for s in scenarios)
    for s in scenarios:
        s["prob"] = round(s["prob"] / total, 2)
    return scenarios, lvls
