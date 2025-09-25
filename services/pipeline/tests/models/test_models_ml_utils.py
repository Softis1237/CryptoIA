from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.models.models_ml import _resolve_horizon_label


def test_resolve_horizon_label_prefers_closest_match():
    per_horizon = {
        "1h": {"horizon_minutes": 60},
        "4h": {"horizon_minutes": 240},
        "24h": {"horizon_minutes": 1440},
    }
    resolved = _resolve_horizon_label(per_horizon, horizon_minutes=70, default_label="4h")
    assert resolved == "1h"


def test_resolve_horizon_label_falls_back_to_default():
    per_horizon = {}
    resolved = _resolve_horizon_label(per_horizon, horizon_minutes=300, default_label="4h")
    assert resolved == "4h"
