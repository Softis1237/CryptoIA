from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
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


class IngestPricesOutput(BaseModel):
    run_id: str
    slot: str
    symbols: List[str]
    start_ts: int
    end_ts: int
    prices_path_s3: str


def run(payload: IngestPricesInput) -> IngestPricesOutput:
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
                {
                    "ts": ts,
                    "open": o,
                    "high": h,
                    "low": low,
                    "close": c,
                    "volume": v,
                    "symbol": symbol,
                }
            )

    df = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume", "symbol"]
    )

    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slot = payload.slot or "manual"
    local_dir = Path(f"runs/{date_key}/{slot}")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / "prices.parquet"

    table = pa.Table.from_pandas(df)
    pq.write_table(table, local_path)

    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    s3_uri = upload_bytes(
        str(local_path),
        sink.getvalue().to_pybytes(),
        content_type="application/octet-stream",
    )

    return IngestPricesOutput(
        run_id=payload.run_id,
        slot=payload.slot,
        symbols=payload.symbols,
        start_ts=payload.start_ts,
        end_ts=payload.end_ts,
        prices_path_s3=s3_uri,
    )


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -m pipeline.data.ingest_prices '<json_payload>'",
            file=sys.stderr,
        )
        raise SystemExit(2)
    payload_raw = sys.argv[1]
    payload = IngestPricesInput.model_validate_json(payload_raw)
    out = run(payload)
    print(out.model_dump_json())


if __name__ == "__main__":
    main()
