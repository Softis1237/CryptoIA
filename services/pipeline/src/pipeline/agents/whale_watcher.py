from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from loguru import logger

from ..data.ingest_onchain import IngestOnchainInput, run as run_onchain
from ..data.ingest_order_flow import IngestOrderFlowInput, run as run_order_flow
from ..infra.db import upsert_strategic_verdict, upsert_whale_details
from ..infra.s3 import download_bytes, upload_bytes

try:  # pragma: no cover - optional heavy deps
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore


@dataclass
class WhaleInput:
    run_id: str
    slot: str = "whale"
    symbol: str = "BTC/USDT"
    provider: str = os.getenv("CCXT_PROVIDER", os.getenv("CCXT_EXCHANGE", "binance"))


def _asset_from_symbol(symbol: str) -> str:
    return symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()


def _fetch_onchain_signals(inp: WhaleInput) -> Dict[str, float]:
    asset = os.getenv("WHALE_ONCHAIN_ASSET", _asset_from_symbol(inp.symbol))
    try:
        out = run_onchain(
            IngestOnchainInput(run_id=f"{inp.run_id}_onchain", slot=inp.slot, asset=asset)
        )
        metrics: Dict[str, float] = {}
        for sig in out.onchain_signals:
            metrics[sig.metric] = sig.value if sig.value is not None else 0.0
        return metrics
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"whale_watcher onchain fetch failed: {exc}")
        return {}


def _extract_large_trades(trades_path: str, threshold_usd: float) -> Dict[str, float]:
    if not trades_path or pa is None or pq is None:
        return {}
    try:
        buf = download_bytes(trades_path)
        table = pq.read_table(pa.BufferReader(buf))
        df = table.to_pandas()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"whale_watcher order flow parquet load failed: {exc}")
        return {}
    if df.empty:
        return {"large_trades": 0, "large_buy": 0, "large_sell": 0}
    try:
        df["price"] = df["price"].astype(float)
        df["amount"] = df["amount"].astype(float)
        df["usd"] = df["price"] * df["amount"]
        sides = df.get("side")
        if sides is not None:
            sides = sides.astype(str).str.lower()
        large_mask = df["usd"] >= threshold_usd
        total = int(large_mask.sum())
        large_buy = int(((sides == "buy") & large_mask).sum()) if sides is not None else 0
        large_sell = int(((sides == "sell") & large_mask).sum()) if sides is not None else 0
        avg_usd = float(df["usd"].mean()) if not df["usd"].empty else 0.0
        med_usd = float(df["usd"].median()) if not df["usd"].empty else 0.0
        buy_vol = float(df.loc[sides == "buy", "amount"].sum()) if sides is not None else 0.0
        sell_vol = float(df.loc[sides == "sell", "amount"].sum()) if sides is not None else 0.0
        return {
            "large_trades": total,
            "large_buy": large_buy,
            "large_sell": large_sell,
            "avg_trade_usd": avg_usd,
            "median_trade_usd": med_usd,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"whale_watcher order flow stats failed: {exc}")
        return {}


def _fetch_order_flow_meta(inp: WhaleInput) -> Dict[str, float]:
    window = int(os.getenv("WHALE_ORDERFLOW_WINDOW_SEC", "180"))
    symbol_ccxt = inp.symbol.replace("/", "")
    try:
        out = run_order_flow(
            IngestOrderFlowInput(
                run_id=f"{inp.run_id}_of",
                slot=inp.slot,
                symbol=symbol_ccxt,
                provider=inp.provider,
                window_sec=window,
            )
        )
        meta: Dict[str, float] = dict(out.meta or {})
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"whale_watcher order flow fetch failed: {exc}")
        return {}
    threshold = float(os.getenv("WHALE_LARGE_TRADE_USD", "250000"))
    stats = _extract_large_trades(out.trades_path_s3, threshold)
    meta.update(stats)
    meta.setdefault("window_sec", window)
    return meta


def _score_mood(onchain: Dict[str, float], order_flow: Dict[str, float]) -> Dict[str, Any]:
    net = float(onchain.get("exchange_netflow") or onchain.get("netflow") or 0.0)
    holders = float(onchain.get("lt_holders", 0.0) or 0.0)
    whale_txs = int(onchain.get("whale_txs", 0) or 0)
    ofi = float(order_flow.get("ofi", 0.0) or 0.0)
    large_buy = int(order_flow.get("large_buy", 0) or 0)
    large_sell = int(order_flow.get("large_sell", 0) or 0)

    net_thr = float(os.getenv("WHALE_NETFLOW_THRESHOLD", "500.0"))
    ofi_thr = float(os.getenv("WHALE_OFI_THRESHOLD", "0.05"))

    score = 0.0
    if net < -abs(net_thr):
        score += 1.0
    elif net > abs(net_thr):
        score -= 1.0

    if ofi > ofi_thr:
        score += 0.5
    elif ofi < -ofi_thr:
        score -= 0.5

    if large_buy > large_sell:
        score += 0.5
    elif large_sell > large_buy:
        score -= 0.5

    if holders > 0:
        score += 0.25
    elif holders < 0:
        score -= 0.25

    if whale_txs > 0 and net < 0:
        score += 0.25
    elif whale_txs > 0 and net > 0:
        score -= 0.25

    if score >= 0.75:
        status = "WHALE_BULLISH"
    elif score <= -0.75:
        status = "WHALE_BEARISH"
    else:
        status = "WHALE_NEUTRAL"

    return {"score": score, "status": status}


def run(inp: WhaleInput) -> Dict[str, Any]:
    onchain = _fetch_onchain_signals(inp)
    order_flow = _fetch_order_flow_meta(inp)
    mood = _score_mood(onchain, order_flow)

    verdict = {
        "status": mood["status"],
        "score": mood["score"],
        "onchain": onchain,
        "order_flow": order_flow,
    }

    now = datetime.now(timezone.utc)
    date_key = now.strftime("%Y-%m-%d")
    path = f"runs/{date_key}/{inp.slot}/whale_verdict.json"
    upload_bytes(path, json.dumps(verdict).encode("utf-8"), content_type="application/json")

    try:
        from ..infra.metrics import push_values

        push_values(
            job="whale_watcher",
            values={"score": float(mood["score"])},
            labels={"symbol": inp.symbol},
        )
    except Exception:
        pass

    try:
        upsert_strategic_verdict(
            agent_name="Whale Watcher",
            symbol=inp.symbol,
            ts=now,
            verdict=str(verdict.get("status")),
            confidence=None,
            meta=verdict,
        )
        exchange_netflow = float(onchain.get("exchange_netflow") or onchain.get("netflow") or 0.0)
        whale_txs = int(onchain.get("whale_txs", 0) or order_flow.get("large_trades", 0) or 0)
        large_trades = int(order_flow.get("large_trades", 0) or whale_txs)
        upsert_whale_details(
            agent_name="Whale Watcher",
            symbol=inp.symbol,
            ts=now,
            exchange_netflow=exchange_netflow,
            whale_txs=whale_txs,
            large_trades=large_trades,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to persist whale verdict: {exc}")
    return verdict


def main() -> None:
    import sys
    import json as _json

    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.agents.whale_watcher '<json_payload>'", file=sys.stderr)
        sys.exit(2)
    payload = WhaleInput(**_json.loads(sys.argv[1]))
    out = run(payload)
    print(_json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
