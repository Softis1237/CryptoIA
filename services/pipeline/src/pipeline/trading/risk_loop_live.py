from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from .exchange import ExchangeClient


@dataclass
class TrailConfig:
    trail_pct: float = 0.01  # 1%
    improve_threshold_pct: float = 0.003  # only move if improves by 0.3%
    interval_s: int = 60


def _trail_stop_for_long(entry: float, high: float, trail_pct: float) -> float:
    return float(high * (1.0 - trail_pct))


def _trail_stop_for_short(entry: float, low: float, trail_pct: float) -> float:
    return float(low * (1.0 + trail_pct))


def risk_loop(symbol: str = os.getenv("EXEC_SYMBOL", "BTC/USDT"), config: Optional[TrailConfig] = None) -> None:
    cfg = config or TrailConfig(
        trail_pct=float(os.getenv("RISK_TRAIL_PCT", "0.01")),
        improve_threshold_pct=float(os.getenv("RISK_IMPROVE_PCT", "0.003")),
        interval_s=int(os.getenv("RISK_LOOP_INTERVAL", "60")),
    )
    ex = ExchangeClient()
    dry = os.getenv("DRY_RUN", "1") in {"1", "true", "True"}
    high = None
    low = None
    while True:
        try:
            px = ex.get_last_price(symbol)
            high = px if high is None else max(high, px)
            low = px if low is None else min(low, px)
            # NOTE: For simplicity we do not introspect positions here; in practice fetch open positions and adjust stop orders.
            # This loop demonstrates how to compute trailing stops and would need exchange-specific mapping for live orders.
            logger.debug(f"risk_loop {symbol}: px={px:.2f} hi={high:.2f} lo={low:.2f}")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"risk_loop error: {e}")
        time.sleep(cfg.interval_s)


def main():
    risk_loop()


if __name__ == "__main__":
    main()

