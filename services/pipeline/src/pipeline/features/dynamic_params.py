from __future__ import annotations

"""Модуль динамического подбора параметров индикаторов."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from pydantic import BaseModel

from ..models.loader import ArtifactNotAvailable, load_model_artifact


class IndicatorConfig(BaseModel):
    """Набор параметров индикаторов для конкретного режима рынка."""

    window_rsi: int = 14
    window_atr: int = 14
    bollinger_window: int = 20
    bollinger_std: float = 2.0


@dataclass(slots=True)
class IndicatorParamModel:
    name: str
    predict_fn: Callable[[np.ndarray], IndicatorConfig]


def infer_regime_label(feature_vec: np.ndarray) -> str:
    """Грубая эвристика режима на основе тренда и волатильности."""

    if feature_vec.size < 2:
        return "range"
    trend, vol = float(feature_vec[0]), float(feature_vec[1])
    if trend > 0.0015 and vol < 0.04:
        return "trend_up"
    if trend < -0.0015 and vol < 0.04:
        return "trend_down"
    if vol > 0.08:
        return "choppy_range"
    return "range"


class RegimeModelRouter:
    """Выбирает ONNX-модель для подбора параметров по текущему режиму."""

    def __init__(self) -> None:
        self._cache: dict[str, IndicatorParamModel] = {}

    def recommend(self, feature_vec: np.ndarray, fallback: IndicatorConfig | None = None) -> IndicatorConfig:
        fallback_cfg = fallback or IndicatorConfig()
        regime = infer_regime_label(feature_vec)
        model = self._load_model(regime)
        if not model:
            return fallback_cfg
        try:
            return model.predict_fn(feature_vec)
        except Exception:
            return fallback_cfg

    @lru_cache(maxsize=8)
    def _load_model(self, regime: str) -> IndicatorParamModel | None:
        try:
            artifact = load_model_artifact(f"indicator_params/{regime}.onnx")
        except (FileNotFoundError, ArtifactNotAvailable):
            return None

        def _predict(features: np.ndarray) -> IndicatorConfig:
            input_name = artifact.get_inputs()[0].name  # type: ignore[call-arg]
            payload = features.astype(np.float32).reshape(1, -1)
            outputs = artifact.run(None, {input_name: payload})
            values = outputs[0][0].tolist()
            return IndicatorConfig(
                window_rsi=int(values[0]),
                window_atr=int(values[1]),
                bollinger_window=int(values[2]),
                bollinger_std=float(values[3]),
            )

        return IndicatorParamModel(name=f"indicator-{regime}", predict_fn=_predict)
