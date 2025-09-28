from __future__ import annotations

from pipeline.agents.advanced_ta_agent import AdvancedTAAgent


def test_advanced_ta_levels_basic():
    base = 100.0
    ohlcv = []
    for i in range(160):
        ohlcv.append(
            {
                "ts": i * 60,
                "open": base + i * 0.5,
                "high": base + i * 0.6,
                "low": base + i * 0.4,
                "close": base + i * 0.55,
            }
        )
    agent = AdvancedTAAgent()
    res = agent.run({"ohlcv": ohlcv, "lookback": 120, "timeframe": "1m"})
    out = res.output
    assert out["symbol"] == "BTC/USDT"
    swing_high = out["swing_high"]
    swing_low = out["swing_low"]
    diff = swing_high - swing_low
    expected_support = round(swing_high - diff * 0.382, 2)
    assert out["fib_support"][0] == expected_support
    assert out["bias"] in {"up_swing", "range_up"}
