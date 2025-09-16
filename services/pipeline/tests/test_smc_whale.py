from __future__ import annotations

import sys
import types

import pytest

from pipeline.agents import smc_analyst as smc
from pipeline.agents import whale_watcher as whale


def _make_ohlcv(prices, low_shift=1.0, high_shift=1.0, volume=100.0):
    rows = []
    for idx, price in enumerate(prices):
        ts = idx * 60
        rows.append(
            [
                ts,
                price - 0.5,
                price + high_shift,
                price - low_shift,
                price,
                volume + idx,
            ]
        )
    return rows


def test_smc_fetch_ohlcv_cache(monkeypatch):
    class FakeExchange:
        fetch_calls = 0

        def __init__(self, config):
            pass

        def fetch_ohlcv(self, symbol, timeframe, limit):  # noqa: D401
            FakeExchange.fetch_calls += 1
            return _make_ohlcv([100.0] * min(limit, 5))

    fake_module = types.SimpleNamespace(binance=FakeExchange)
    monkeypatch.setitem(sys.modules, "ccxt", fake_module)
    monkeypatch.setenv("SMC_OHLC_CACHE_SEC", "600")
    smc._reset_cache()
    smc._fetch_ohlcv("BTC/USDT", "binance", "1h")
    smc._fetch_ohlcv("BTC/USDT", "binance", "1h")
    assert FakeExchange.fetch_calls == 1


def test_smc_run_bullish_setup(monkeypatch):
    prices_4h = [120.0 for _ in range(200)]
    prices_15m = [110.0 for _ in range(199)] + [108.0]
    prices_5m = [100.0 + 0.4 * i for i in range(200)]

    data_4h = _make_ohlcv(prices_4h, low_shift=20.0, high_shift=5.0)
    data_15m = _make_ohlcv(prices_15m, low_shift=12.0, high_shift=3.0)
    data_5m = _make_ohlcv(prices_5m, low_shift=4.0, high_shift=4.0)
    # inject a strong bullish candle into 5m data to create POI
    ts, _, _, _, _, vol = data_5m[-5]
    data_5m[-5] = [ts, 100.0, 118.0, 95.0, 112.0, vol]
    # ensure last closes show clear bullish CHOCH
    for offset in range(1, 6):
        ts2, _, _, _, _, vol2 = data_5m[-offset]
        level = 105.0 + (5 - offset) * 1.4
        data_5m[-offset] = [ts2, level - 2.0, level + 2.5, level - 3.0, level, vol2]

    def fake_fetch(symbol: str, provider: str, timeframe: str, limit: int = 500):
        if timeframe == "4h":
            return data_4h
        if timeframe == "15m":
            return data_15m
        if timeframe == "5m":
            return data_5m
        raise AssertionError(f"Unexpected timeframe {timeframe}")

    smc._reset_cache()
    monkeypatch.setattr(smc, "_fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(smc, "upload_bytes", lambda *args, **kwargs: "s3://test")
    stored = {}

    def fake_upsert(agent_name, symbol, ts, verdict, confidence, meta):  # noqa: D401
        stored["verdict"] = verdict
        stored["meta"] = meta

    monkeypatch.setattr(smc, "upsert_strategic_verdict", fake_upsert)
    out = smc.run(smc.SMCInput(run_id="test", symbol="BTC/USDT"))
    assert out["status"] == "SMC_BULLISH_SETUP"
    assert stored["verdict"] == "SMC_BULLISH_SETUP"
    assert out["entry_zone"] is not None


def test_whale_watcher_scoring(monkeypatch):
    monkeypatch.setattr(
        whale,
        "_fetch_onchain_signals",
        lambda inp: {"exchange_netflow": -1200.0, "whale_txs": 4, "lt_holders": 0.3},
    )
    monkeypatch.setattr(
        whale,
        "_fetch_order_flow_meta",
        lambda inp: {"ofi": 0.2, "large_buy": 5, "large_sell": 1, "large_trades": 6},
    )
    monkeypatch.setattr(whale, "upload_bytes", lambda *args, **kwargs: "s3://test")
    stored = {}

    def fake_upsert(agent_name, symbol, ts, verdict, confidence, meta):  # noqa: D401
        stored["verdict"] = verdict
        stored["meta"] = meta

    monkeypatch.setattr(whale, "upsert_strategic_verdict", fake_upsert)
    monkeypatch.setattr(whale, "upsert_whale_details", lambda *args, **kwargs: None)
    out = whale.run(whale.WhaleInput(run_id="t", symbol="BTC/USDT"))
    assert out["status"] == "WHALE_BULLISH"
    assert stored["verdict"] == "WHALE_BULLISH"
    assert out["score"] > 0
