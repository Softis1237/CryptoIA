from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from pydantic import BaseModel, Field

from ..infra.s3 import upload_bytes
from ..utils.ccxt_helpers import fetch_ohlcv as ccxt_fetch_ohlcv


class IngestPricesInput(BaseModel):
    run_id: str
    slot: str
    symbols: List[str]
    start_ts: int
    end_ts: int
    provider: str | None = None
    fallback_providers: List[str] = Field(default_factory=list)
    timeout_ms: int | None = None
    retries: int | None = None
    retry_backoff_sec: float | None = None


class IngestPricesOutput(BaseModel):
    run_id: str
    slot: str
    symbols: List[str]
    start_ts: int
    end_ts: int
    prices_path_s3: str


def run(payload: IngestPricesInput) -> IngestPricesOutput:
    default_provider = os.getenv("CCXT_EXCHANGE", os.getenv("CCXT_PROVIDER", "binance"))
    providers: List[str] = []
    if payload.provider:
        providers.append(payload.provider)
    fallback_cfg = list(payload.fallback_providers or [])
    env_fallback = os.getenv("PRICES_FALLBACK_PROVIDERS", "")
    if env_fallback:
        fallback_cfg.extend([p.strip() for p in env_fallback.split(",") if p.strip()])
    for prov in fallback_cfg:
        if prov and prov not in providers:
            providers.append(prov)
    if default_provider and default_provider not in providers:
        providers.append(default_provider)
    timeout_ms = int(payload.timeout_ms or os.getenv("CCXT_TIMEOUT_MS", "20000"))
    retries = int(payload.retries or os.getenv("CCXT_RETRIES", "3"))
    backoff = float(payload.retry_backoff_sec or os.getenv("CCXT_RETRY_BACKOFF_SEC", "1.0"))

    rows = []
    for symbol in payload.symbols:
        # ccxt expects milliseconds; convert once
        since_ms = int(payload.start_ts) * 1000
        end_ms = int(payload.end_ts) * 1000
        # 1m candles count (add small buffer)
        limit = int((payload.end_ts - payload.start_ts) / 60) + 10
        try:
            ohlcv = ccxt_fetch_ohlcv(
                providers,
                symbol,
                "1m",
                limit,
                since_ms=since_ms,
                timeout_ms=timeout_ms,
                retries=retries,
                backoff=backoff,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"ingest_prices failed for {symbol}: {exc}")
            raise
        for ts, o, h, low, c, v in ohlcv:
            if ts > end_ms:
                break
            rows.append(
                {
                    "ts": int(ts // 1000),
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
