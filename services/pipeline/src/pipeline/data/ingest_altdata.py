from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List

import ccxt  # type: ignore
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from pydantic import BaseModel

from ..infra.s3 import upload_bytes


class IngestAltDataInput(BaseModel):
    run_id: str
    slot: str
    symbol: str = "BTC/USDT"


class IngestAltDataOutput(BaseModel):
    run_id: str
    slot: str
    alt_path_s3: str
    metrics: List[dict]


def _parse_exchanges() -> List[str]:
    raw = os.getenv("ALT_EXCHANGES", "binance,bybit,okx,kraken")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _slippage_bps_for_orderbook(ob, side: str, order_size_usd: float, price_hint: float) -> float:
    try:
        if not ob or not ob.get(side):
            return 0.0
        remain = float(order_size_usd)
        ref = float(price_hint) if price_hint else 0.0
        if ref <= 0:
            return 0.0
        lvl_list = ob[side]
        filled = 0.0
        cost = 0.0
        for price, amount in lvl_list:
            price = float(price)
            amount = float(amount)
            lvl_value = price * amount
            if lvl_value >= remain:
                need_amt = remain / max(1e-9, price)
                cost += need_amt * price
                filled += need_amt
                remain = 0.0
                break
            else:
                cost += lvl_value
                filled += amount
                remain -= lvl_value
        if filled <= 0.0:
            return 0.0
        avg_exec_price = cost / max(1e-9, filled)
        slip = (avg_exec_price - ref) / ref if side == "asks" else (ref - avg_exec_price) / ref
        return max(0.0, float(slip) * 1e4)
    except Exception:
        return 0.0


def run(payload: IngestAltDataInput) -> IngestAltDataOutput:
    exchanges = _parse_exchanges()
    order_usd = float(os.getenv("ALT_ORDER_SIZE_USD", "100000"))
    symbol = payload.symbol

    metrics: List[dict] = []
    vol_sum_usd = 0.0
    slips: List[float] = []
    now = int(datetime.now(timezone.utc).timestamp())

    for ex_name in exchanges:
        try:
            ex = getattr(ccxt, ex_name)({"enableRateLimit": True, "timeout": 15000})
            ticker = ex.fetch_ticker(symbol)
            last = float(ticker.get("last") or ticker.get("close") or 0.0)
            vol = float(ticker.get("quoteVolume") or 0.0)
            vol_sum_usd += vol
            ob = ex.fetch_order_book(symbol, limit=50)
            slip_buy = _slippage_bps_for_orderbook(ob, "asks", order_usd, last)
            slip_sell = _slippage_bps_for_orderbook(ob, "bids", order_usd, last)
            slip_avg = (slip_buy + slip_sell) / 2.0
            slips.append(slip_avg)
        except Exception as e:
            logger.warning(f"altdata: failed for {ex_name}: {e}")
            continue

    metrics.append({"metric": "liquidity_total_24h_usd", "value": vol_sum_usd, "ts": now})
    if slips:
        metrics.append({"metric": "slippage_bps_avg_all", "value": float(sum(slips) / len(slips)), "ts": now})
    for i, ex_name in enumerate(exchanges[:5]):
        if i < len(slips):
            metrics.append({"metric": f"slippage_bps_avg_{ex_name}", "value": float(slips[i]), "ts": now})

    df = pd.DataFrame(metrics)
    table = pa.Table.from_pandas(df)
    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    s3_path = f"runs/{date_key}/{payload.slot}/altdata.parquet"
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    s3_uri = upload_bytes(s3_path, sink.getvalue().to_pybytes(), content_type="application/octet-stream")

    return IngestAltDataOutput(run_id=payload.run_id, slot=payload.slot, alt_path_s3=s3_uri, metrics=metrics)

