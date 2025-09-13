from __future__ import annotations

import os
from datetime import datetime, timezone
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
    provider_name = os.getenv("CCXT_EXCHANGE", os.getenv("CCXT_PROVIDER", "binance"))
    timeout_ms = int(os.getenv("CCXT_TIMEOUT_MS", "20000"))
    ex_cls = getattr(ccxt, provider_name)
    ex = ex_cls({"enableRateLimit": True, "timeout": timeout_ms})

    rows = []
    retries = int(os.getenv("CCXT_RETRIES", "3"))
    backoff = float(os.getenv("CCXT_RETRY_BACKOFF_SEC", "1.0"))
    for symbol in payload.symbols:
        since = payload.start_ts
        limit = int((payload.end_ts - payload.start_ts) / 60000) + 10
        attempt = 0
        while True:
            try:
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
                break
            except Exception:
                attempt += 1
                if attempt > retries:
                    raise
                import time
                time.sleep(backoff * attempt)

    df = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume", "symbol"]
    )

    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slot = payload.slot or "manual"
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    s3_key = f"runs/{date_key}/{slot}/prices.parquet"
    s3_uri = upload_bytes(s3_key, sink.getvalue().to_pybytes(), content_type="application/octet-stream")

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
