from __future__ import annotations

"""Заготовки для симуляционных сценариев Red Team."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ...trading.backtest.types import BacktestConfig


@dataclass(slots=True)
class RedTeamScenario:
    name: str
    description: str
    parameters: Dict[str, float]
    data: pd.DataFrame
    config: BacktestConfig
    strategy_path: str

    def to_payload(self) -> Dict[str, float]:
        return dict(self.parameters)


def generate_price_path(
    base_price: float,
    stress_factor: float,
    length: int,
    bias: str,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = max(120, length)
    timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=steps, freq="5min")
    drift = 0.0
    if bias == "bullish":
        drift = 0.0005 * stress_factor
    elif bias == "bearish":
        drift = -0.0005 * stress_factor
    noise = rng.normal(loc=0.0, scale=0.003 * stress_factor, size=steps)
    returns = drift + noise
    prices = np.maximum(base_price * np.exp(np.cumsum(returns)), 1.0)
    highs = prices * (1 + rng.uniform(0, 0.001 * stress_factor, size=steps))
    lows = prices * (1 - rng.uniform(0, 0.001 * stress_factor, size=steps))
    opens = np.concatenate([[base_price], prices[:-1]])
    volumes = rng.uniform(100, 500, size=steps) * stress_factor
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": np.maximum(highs, prices),
            "low": np.minimum(lows, prices),
            "close": prices,
            "volume": volumes,
        }
    )
    return df


def build_scenario_from_lesson(
    lesson: Dict,
    stress_factor: float,
    strategy_path: str,
    base_price: float = 30_000.0,
) -> RedTeamScenario:
    bias = str(lesson.get("planned_bias") or lesson.get("triggering_signals", ["neutral"])[:1]).lower()
    if "bear" in bias:
        direction = "bearish"
    elif "bull" in bias:
        direction = "bullish"
    else:
        direction = "neutral"
    df = generate_price_path(base_price, stress_factor, 240, direction)
    cfg = BacktestConfig(symbol=str(lesson.get("symbol", "BTCUSDT")))
    return RedTeamScenario(
        name=f"stress_{lesson.get('error_type', 'unknown')}",
        description=lesson.get("correct_action_suggestion", ""),
        parameters={"stress_factor": stress_factor},
        data=df,
        config=cfg,
        strategy_path=strategy_path,
    )
