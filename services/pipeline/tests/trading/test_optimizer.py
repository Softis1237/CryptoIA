from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.trading.optimizer import Signal, build_adaptive_risk_profile


def test_build_adaptive_risk_profile_short_horizon():
    signal = Signal(name="test-short", proba_up=0.68, rr=1.8, atr=80.0, price=50_000.0)
    profile = build_adaptive_risk_profile(
        signal,
        forecast_horizon="45m",
        leverage_cap=20.0,
        historical_target_minutes=[18, 26, 28, 24],
        historical_stop_minutes=[22, 24, 21],
        min_hold_minutes=10,
        min_valid_minutes=10,
    )

    assert 10 <= profile.hold_minutes <= 60
    assert profile.valid_minutes <= profile.hold_minutes
    assert profile.sl_atr_multiplier < 1.2
    assert profile.leverage_factor > 0.75
    assert profile.risk_multiplier >= 1.0
    assert any(note.startswith("bucket:ultra_short") for note in profile.notes)


def test_build_adaptive_risk_profile_long_horizon():
    signal = Signal(name="test-long", proba_up=0.58, rr=2.4, atr=150.0, price=48_000.0)
    profile = build_adaptive_risk_profile(
        signal,
        forecast_horizon="12h",
        leverage_cap=15.0,
        historical_target_minutes=[540, 600, 720, 660],
        historical_stop_minutes=[420, 480, 510],
        min_hold_minutes=60,
        min_valid_minutes=45,
    )

    assert profile.hold_minutes >= 420
    assert profile.sl_atr_multiplier >= 1.8
    assert profile.leverage_factor <= 0.7
    assert profile.risk_multiplier <= 1.0
    assert profile.valid_minutes <= profile.hold_minutes
    assert any(note.startswith("bucket:positional") for note in profile.notes)
