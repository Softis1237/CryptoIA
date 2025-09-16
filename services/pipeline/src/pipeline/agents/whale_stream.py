from __future__ import annotations

"""Whale Watcher stream skeleton (always-on).

Listens to large trades (and optionally on-chain webhooks) and upserts verdicts.

Notes:
- For production use, integrate a proper WS client (ccxt.pro or exchange-native SDK) and Dune/webhook.
- This skeleton falls back to short REST polling if WS is unavailable.
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from ..infra.db import upsert_strategic_verdict, upsert_whale_details
from ..infra.health import start_background as start_health_server


@dataclass
class StreamInput:
    symbol: str = os.getenv("STREAM_SYMBOL", os.getenv("PAPER_SYMBOL", "BTC/USDT"))
    provider: str = os.getenv("CCXT_EXCHANGE", os.getenv("CCXT_PROVIDER", "binance"))
    large_trade_usd: float = float(os.getenv("WHALE_TRADE_USD", "250000"))  # 250k USD
    poll_sec: float = float(os.getenv("WHALE_POLL_SEC", "5"))


async def _poll_loop(inp: StreamInput) -> None:
    import ccxt  # type: ignore

    ex = getattr(ccxt, inp.provider)({"enableRateLimit": True, "timeout": 20000})
    market = inp.symbol
    last_ts = 0
    while True:
        try:
            trades = ex.fetch_trades(market, since=last_ts or None, limit=100)
            now = datetime.now(timezone.utc)
            large_count = 0
            netflow = 0.0
            px = None
            for t in trades or []:
                ts = int(t.get("timestamp") or 0)
                if ts <= last_ts:
                    continue
                last_ts = max(last_ts, ts)
                price = float(t.get("price", 0.0) or 0.0)
                amount = float(t.get("amount", 0.0) or 0.0)
                side = str(t.get("side", ""))
                px = price
                usd = price * amount
                if usd >= inp.large_trade_usd:
                    large_count += 1
                netflow += (amount if side == "buy" else -amount)
            if large_count > 0:
                verdict = "WHALE_BULLISH" if netflow > 0 else ("WHALE_BEARISH" if netflow < 0 else "WHALE_NEUTRAL")
                upsert_strategic_verdict(
                    agent_name="Whale Watcher",
                    symbol=inp.symbol,
                    ts=now,
                    verdict=verdict,
                    confidence=None,
                    meta={"px": px, "large_trades": large_count, "netflow": netflow},
                )
                upsert_whale_details(
                    agent_name="Whale Watcher",
                    symbol=inp.symbol,
                    ts=now,
                    exchange_netflow=float(netflow),
                    whale_txs=int(large_count),
                    large_trades=int(large_count),
                )
            # Push metrics
            try:
                from ..infra.metrics import push_values
                push_values(job="whale_stream", values={"large_trades": float(large_count), "netflow": float(netflow)}, labels={"symbol": inp.symbol})
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001
            logger.debug(f"whale poll error: {e}")
        await asyncio.sleep(max(1.0, inp.poll_sec))


async def _ws_loop(inp: StreamInput) -> None:
    # Optional WS using ccxt.pro (if installed and CCXT_PRO=1)
    try:
        import ccxt.pro as ccxtpro  # type: ignore
    except Exception as e:
        raise RuntimeError("ccxt.pro not available")
    ex = getattr(ccxtpro, inp.provider)({"enableRateLimit": True, "newUpdates": True})
    symbol = inp.symbol
    last_push = 0
    try:
        while True:
            try:
                trades = await ex.watch_trades(symbol)
                now = datetime.now(timezone.utc)
                large = 0
                net = 0.0
                px = None
                for t in trades or []:
                    price = float(t.get("price", 0.0) or 0.0)
                    amount = float(t.get("amount", 0.0) or 0.0)
                    side = str(t.get("side", ""))
                    px = price
                    usd = price * amount
                    if usd >= inp.large_trade_usd:
                        large += 1
                    net += (amount if side == "buy" else -amount)
                if large > 0:
                    verdict = "WHALE_BULLISH" if net > 0 else ("WHALE_BEARISH" if net < 0 else "WHALE_NEUTRAL")
                    upsert_strategic_verdict("Whale Watcher", symbol, now, verdict, None, {"px": px, "large_trades": large, "netflow": net})
                    upsert_whale_details("Whale Watcher", symbol, now, float(net), int(large), int(large))
                # Push metrics throttled
                if (now.timestamp() - last_push) > 5:
                    try:
                        from ..infra.metrics import push_values
                        push_values(job="whale_stream", values={"large_trades": float(large), "netflow": float(net)}, labels={"symbol": symbol})
                    except Exception:
                        pass
                    last_push = now.timestamp()
            except Exception as e:
                logger.debug(f"whale ws error: {e}")
                await asyncio.sleep(1.0)
    finally:
        try:
            await ex.close()
        except Exception:
            pass


def main() -> None:
    inp = StreamInput()
    start_health_server()
    try:
        if os.getenv("CCXT_PRO", "0") in {"1", "true", "True"}:
            asyncio.run(_ws_loop(inp))
        else:
            asyncio.run(_poll_loop(inp))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
