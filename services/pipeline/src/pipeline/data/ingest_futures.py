from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from ..infra.s3 import upload_bytes


class FuturesMeta(BaseModel):
    ts: int
    funding_rate: Optional[float] = None
    mark_price: Optional[float] = None
    open_interest: Optional[float] = None


class IngestFuturesInput(BaseModel):
    run_id: str
    slot: str
    symbol: str = "BTC/USDT"


class IngestFuturesOutput(BaseModel):
    run_id: str
    slot: str
    symbol: str
    futures_meta: FuturesMeta
    futures_path_s3: str


def run(payload: IngestFuturesInput) -> IngestFuturesOutput:
    provider: str = (
        os.getenv("CCXT_FUTURES_PROVIDER")
        or os.getenv("CCXT_PROVIDER")
        or "binanceusdm"
    )
    timeout_ms = int(os.getenv("CCXT_TIMEOUT_MS", "20000"))
    ex = getattr(ccxt, provider)({"enableRateLimit": True, "timeout": timeout_ms})
    now = datetime.now(timezone.utc)
    ts = int(now.timestamp())

    funding_rate: Optional[float] = None
    mark_price: Optional[float] = None
    open_interest: Optional[float] = None

    retries = int(os.getenv("CCXT_RETRIES", "3"))
    backoff = float(os.getenv("CCXT_RETRY_BACKOFF_SEC", "1.0"))
    import time
    # funding rate + mark
    attempt = 0
    while True:
        try:
            fr = ex.fetch_funding_rate(payload.symbol)
            funding_rate = (
                float(fr.get("fundingRate"))
                if fr and fr.get("fundingRate") is not None
                else None
            )
            mark_price = (
                float(fr.get("markPrice"))
                if fr and fr.get("markPrice") is not None
                else None
            )
            break
        except Exception:
            attempt += 1
            if attempt > retries:
                break
            time.sleep(backoff * attempt)
    # open interest
    attempt = 0
    while True:
        try:
            oi = ex.fetch_open_interest(payload.symbol)
            open_interest = (
                float(oi.get("openInterest"))
                if oi and oi.get("openInterest") is not None
                else None
            )
            break
        except Exception:
            attempt += 1
            if attempt > retries:
                break
            time.sleep(backoff * attempt)

    meta = FuturesMeta(
        ts=ts,
        funding_rate=funding_rate,
        mark_price=mark_price,
        open_interest=open_interest,
    )

    df = pd.DataFrame([meta.model_dump()])
    table = pa.Table.from_pandas(df)
    date_key = now.strftime("%Y-%m-%d")
    s3_path = f"runs/{date_key}/{payload.slot}/futures.parquet"
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    s3_uri = upload_bytes(
        s3_path, sink.getvalue().to_pybytes(), content_type="application/octet-stream"
    )

    return IngestFuturesOutput(
        run_id=payload.run_id,
        slot=payload.slot,
        symbol=payload.symbol,
        futures_meta=meta,
        futures_path_s3=s3_uri,
    )
