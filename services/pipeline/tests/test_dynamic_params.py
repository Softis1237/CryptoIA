from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.features.dynamic_params import IndicatorConfig, RegimeModelRouter, infer_regime_label


def test_infer_regime_label_handles_trend_variations() -> None:
    vec_up = np.array([0.002, 0.02, 0.0])
    vec_down = np.array([-0.002, 0.02, 0.0])
    vec_range = np.array([0.0, 0.03, 0.0])
    assert infer_regime_label(vec_up) == "trend_up"
    assert infer_regime_label(vec_down) == "trend_down"
    assert infer_regime_label(vec_range) == "range"


def test_router_returns_fallback_when_artifact_missing(monkeypatch) -> None:
    router = RegimeModelRouter()
    vec = np.array([0.001, 0.05, 0.3], dtype=float)
    fallback = IndicatorConfig(window_rsi=21, window_atr=18, bollinger_window=34, bollinger_std=2.5)
    def _raise(*args, **kwargs):
        raise FileNotFoundError
    monkeypatch.setattr("pipeline.features.dynamic_params.load_model_artifact", _raise)
    cfg = router.recommend(vec, fallback=fallback)
    assert cfg.window_rsi == fallback.window_rsi
    assert cfg.bollinger_std == fallback.bollinger_std
