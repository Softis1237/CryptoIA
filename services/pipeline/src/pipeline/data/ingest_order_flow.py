from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import ccxt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from ..infra.s3 import upload_bytes


class IngestOrderFlowInput(BaseModel):
    run_id: str
    slot: str
    symbol: str = "BTCUSDT"  # exchange-specific (binance: BTCUSDT)
    provider: str = "binance"
    window_sec: int = 60


class IngestOrderFlowOutput(BaseModel):
    run_id: str
    slot: str
    symbol: str
    trades_path_s3: str
    meta: Dict[str, float]


def _to_ccxt_symbol(sym: str) -> str:
    if "/" in sym:
        return sym
    try:
        base, quote = sym[:-4], sym[-4:]
        # naive split for USDT pairs
        if sym.endswith("USDT"):
            base, quote = sym[:-4], "USDT"
        elif sym.endswith("USD"):
            base, quote = sym[:-3], "USD"
        return f"{base}/{quote}"
    except Exception:
        return sym


def _fetch_trades_rest(ex, market: str, since_ms: Optional[int], limit: int = 1000) -> List[dict]:
    try:
        trades = ex.fetch_trades(market, since=since_ms, limit=limit)
        return trades or []
    except Exception:
        return []


def run(payload: IngestOrderFlowInput) -> IngestOrderFlowOutput:
    provider = payload.provider or os.getenv("FUTURES_PROVIDER", os.getenv("CCXT_PROVIDER", "binance"))
    timeout_ms = int(os.getenv("CCXT_TIMEOUT_MS", "20000"))
    ex = getattr(ccxt, provider)({"enableRateLimit": True, "timeout": timeout_ms})
    market = _to_ccxt_symbol(payload.symbol)

    rows: List[dict] = []
    use_ws = os.getenv("ORDERFLOW_USE_WS", "0") in {"1", "true", "True"}
    if use_ws:
        try:
            import asyncio
            import ccxt.pro as ccxtpro  # type: ignore

            async def _collect_ws() -> List[dict]:
                out: List[dict] = []
                seen: set[str] = set()
                client = getattr(ccxtpro, provider)({"enableRateLimit": True})
                t_end = time.time() + max(1, int(payload.window_sec))
                try:
                    while time.time() < t_end:
                        batch = await client.watch_trades(market)
                        if not batch:
                            continue
                        if isinstance(batch, dict):
                            batch = [batch]
                        for t in batch:
                            tid = str(t.get("id") or f"{t.get('timestamp')}-{t.get('price')}-{t.get('amount')}")
                            if tid in seen:
                                continue
                            seen.add(tid)
                            ts_ms = int(t.get("timestamp") or 0)
                            price = float(t.get("price") or 0.0)
                            amount = float(t.get("amount") or 0.0)
                            side = str(t.get("side") or "")
                            out.append({"ts": ts_ms // 1000, "price": price, "amount": amount, "side": side})
                finally:
                    try:
                        await client.close()
                    except Exception:
                        pass
                return out

            rows = asyncio.run(_collect_ws())
        except Exception:
            # fallback to REST
            use_ws = False

    if not use_ws:
        # REST fallback loop for window_sec seconds
        start = time.time()
        since_ms: Optional[int] = None
        seen_ids: set[str] = set()
        poll_int = float(os.getenv("ORDERFLOW_POLL_SEC", "1.0"))
        while (time.time() - start) < max(1, int(payload.window_sec)):
            trades = _fetch_trades_rest(ex, market, since_ms)
            if trades:
                for t in trades:
                    tid = str(t.get("id") or f"{t.get('timestamp')}-{t.get('price')}-{t.get('amount')}")
                    if tid in seen_ids:
                        continue
                    seen_ids.add(tid)
                    ts_ms = int(t.get("timestamp") or 0)
                    price = float(t.get("price") or 0.0)
                    amount = float(t.get("amount") or 0.0)
                    side = str(t.get("side") or "")
                    rows.append({"ts": ts_ms // 1000, "price": price, "amount": amount, "side": side})
                since_ms = int(trades[-1].get("timestamp") or since_ms or 0)
            time.sleep(max(0.05, poll_int))

    if not rows:
        # ensure non-empty parquet schema
        rows = [{"ts": int(time.time()), "price": 0.0, "amount": 0.0, "side": ""}]

    df = pd.DataFrame(rows, columns=["ts", "price", "amount", "side"]).sort_values("ts")

    # Meta summary
    buy_vol = float(df.loc[df["side"] == "buy", "amount"].sum()) if "side" in df.columns else 0.0
    sell_vol = float(df.loc[df["side"] == "sell", "amount"].sum()) if "side" in df.columns else 0.0
    total_vol = float(df["amount"].sum())
    ofi = (buy_vol - sell_vol) / total_vol if total_vol else 0.0
    meta = {
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "total_vol": total_vol,
        "ofi": ofi,
        "n_trades": int(len(df)),
        "window_sec": int(payload.window_sec),
    }

    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    buf = sink.getvalue().to_pybytes()

    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slot = payload.slot or "manual"
    s3_path = f"runs/{date_key}/{slot}/trades_{payload.symbol}.parquet"
    s3_uri = upload_bytes(s3_path, buf, content_type="application/octet-stream")

    return IngestOrderFlowOutput(
        run_id=payload.run_id,
        slot=payload.slot,
        symbol=payload.symbol,
        trades_path_s3=s3_uri,
        meta=meta,
    )


def main():
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -m pipeline.data.ingest_order_flow '<json_payload>'",
            file=sys.stderr,
        )
        raise SystemExit(2)
    payload_raw = sys.argv[1]
    payload = IngestOrderFlowInput.model_validate_json(payload_raw)
    out = run(payload)
    print(out.model_dump_json())


if __name__ == "__main__":
    main()
