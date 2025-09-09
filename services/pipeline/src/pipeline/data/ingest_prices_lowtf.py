from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import List

from loguru import logger
from pydantic import BaseModel

from ..infra.s3 import upload_bytes


class IngestPricesLowTFInput(BaseModel):
    run_id: str
    slot: str
    symbol: str
    start_ts: int
    end_ts: int
    timeframe_seconds: int


class IngestPricesLowTFOutput(BaseModel):
    prices_lowtf_path_s3: str


def _fetch_ohlcv(
    exchange, symbol: str, timeframe: str, since_ms: int, end_ms: int, step_ms: int
) -> List[List[float]]:
    """Fetch OHLCV data in a loop until end_ms or no more data."""
    all_ohlcv: List[List[float]] = []
    while since_ms < end_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=1000
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"fetch_ohlcv failed: {e}")
            break
        if not batch:
            break
        all_ohlcv.extend(batch)
        last = batch[-1][0]
        since_ms = last + step_ms
        if last >= end_ms:
            break
    return all_ohlcv


def _aggregate_from_trades(
    exchange, symbol: str, timeframe_seconds: int, start_ms: int, end_ms: int
):
    import pandas as pd  # type: ignore[import-not-found]

    trades = []
    since = start_ms
    while since < end_ms:
        try:
            batch = exchange.fetch_trades(symbol, since=since, limit=1000)
        except Exception as e:  # noqa: BLE001
            logger.error(f"fetch_trades failed: {e}")
            break
        if not batch:
            break
        trades.extend(batch)
        since = batch[-1]["timestamp"] + 1
    if not trades:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    ohlcv = df.resample(f"{timeframe_seconds}S").agg(
        {"price": ["first", "max", "min", "last"], "amount": "sum"}
    )
    ohlcv.columns = ["open", "high", "low", "close", "volume"]
    ohlcv.dropna(inplace=True)
    ohlcv["ts"] = ohlcv.index.astype("int64") // 10**9
    return ohlcv.reset_index(drop=True)


def run(payload: IngestPricesLowTFInput) -> IngestPricesLowTFOutput:
    import ccxt  # type: ignore[import-not-found]
    import pandas as pd  # type: ignore[import-not-found]
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    exchange = ccxt.binance({"enableRateLimit": True})
    timeframe = f"{payload.timeframe_seconds}s"
    start_ms = payload.start_ts * 1000
    end_ms = payload.end_ts * 1000
    step_ms = payload.timeframe_seconds * 1000

    ohlcv = _fetch_ohlcv(exchange, payload.symbol, timeframe, start_ms, end_ms, step_ms)

    if ohlcv:
        df = pd.DataFrame(
            ohlcv, columns=["ts_ms", "open", "high", "low", "close", "volume"]
        )
        df = df[(df["ts_ms"] >= start_ms) & (df["ts_ms"] <= end_ms)]
        df["ts"] = (df["ts_ms"] // 1000).astype(int)
        df = df[["ts", "open", "high", "low", "close", "volume"]]
    else:
        logger.info("No OHLCV returned, aggregating from trades")
        df = _aggregate_from_trades(
            exchange, payload.symbol, payload.timeframe_seconds, start_ms, end_ms
        )

    df = df.sort_values("ts").reset_index(drop=True)

    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slot = payload.slot or "manual"
    s3_path = f"runs/{date_key}/{slot}/prices_lowtf.parquet"

    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    buf = sink.getvalue().to_pybytes()
    s3_uri = upload_bytes(s3_path, buf, content_type="application/octet-stream")

    return IngestPricesLowTFOutput(prices_lowtf_path_s3=s3_uri)


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m pipeline.data.ingest_prices_lowtf '<json_payload>'",
            file=sys.stderr,
        )
        sys.exit(2)
    payload_raw = sys.argv[1]
    payload = IngestPricesLowTFInput.model_validate_json(payload_raw)
    out = run(payload)
    print(out.model_dump_json())


if __name__ == "__main__":
    main()
