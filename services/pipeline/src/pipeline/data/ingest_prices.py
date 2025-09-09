from __future__ import annotations

import io
import os
import re
from typing import List

import ccxt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from ..infra.s3 import upload_bytes


class IngestPricesInput(BaseModel):
    run_id: str
    slot: str
    symbols: List[str]
    start_ts: int
    end_ts: int
    prices_path_s3: str


class IngestPricesOutput(BaseModel):
    run_id: str
    slot: str
    symbols: List[str]
    start_ts: int
    end_ts: int
    prices_path_s3: str


def _validate_path(path: str, slot: str) -> None:
    pattern = rf"^runs/\d{{4}}-\d{{2}}-\d{{2}}/{re.escape(slot)}/prices\.parquet$"
    if not re.match(pattern, path):
        raise ValueError(
            "prices_path_s3 must match runs/{YYYY-MM-DD}/{slot}/prices.parquet"
        )


def run(payload: IngestPricesInput) -> IngestPricesOutput:
    _validate_path(payload.prices_path_s3, payload.slot)

    provider_name = os.getenv("CCXT_PROVIDER", "binance")
    ex_cls = getattr(ccxt, provider_name)
    ex = ex_cls({"enableRateLimit": True})

    rows = []
    for symbol in payload.symbols:
        since = payload.start_ts
        limit = int((payload.end_ts - payload.start_ts) / 60000) + 10
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1m", since=since, limit=limit)
        for ts, o, h, low, c, v in ohlcv:
            if ts > payload.end_ts:
                break
            rows.append(
                {"ts": ts, "open": o, "high": h, "low": low, "close": c, "volume": v}
            )

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    s3_uri = upload_bytes(
        payload.prices_path_s3, buf.getvalue(), content_type="application/parquet"
    )

    return IngestPricesOutput(**payload.dict(), prices_path_s3=s3_uri)
