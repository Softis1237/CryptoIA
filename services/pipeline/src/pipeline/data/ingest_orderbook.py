from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import ccxt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from ..infra.s3 import upload_bytes


class IngestOrderbookInput(BaseModel):
    run_id: str
    slot: str
    symbol: str
    depth: int = 20


class IngestOrderbookOutput(BaseModel):
    run_id: str
    slot: str
    symbol: str
    depth: int
    book_path_s3: str
    meta: Dict[str, Any]


def run(payload: IngestOrderbookInput) -> IngestOrderbookOutput:
    exchange = ccxt.binance()
    ob = exchange.fetch_order_book(payload.symbol, limit=payload.depth)
    bids: List[List[float]] = ob.get("bids", [])[: payload.depth]
    asks: List[List[float]] = ob.get("asks", [])[: payload.depth]

    bid_liq = sum(b[1] for b in bids)
    ask_liq = sum(a[1] for a in asks)
    total_liq = bid_liq + ask_liq
    imbalance = (bid_liq - ask_liq) / total_liq if total_liq else 0.0
    depth_ratio = bid_liq / ask_liq if ask_liq else 0.0

    meta = {
        "imbalance": float(imbalance),
        "depth_ratio": float(depth_ratio),
        "total_liq": float(total_liq),
        "bid_levels": len(bids),
        "ask_levels": len(asks),
    }

    df_b = pd.DataFrame(bids, columns=["price", "amount"])
    df_b["side"] = "bid"
    df_a = pd.DataFrame(asks, columns=["price", "amount"])
    df_a["side"] = "ask"
    df = pd.concat([df_b, df_a], ignore_index=True)

    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    buf = sink.getvalue().to_pybytes()

    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slot = payload.slot or "manual"
    s3_path = f"runs/{date_key}/{slot}/orderbook_{payload.symbol}.parquet"
    s3_uri = upload_bytes(s3_path, buf, content_type="application/octet-stream")

    return IngestOrderbookOutput(
        run_id=payload.run_id,
        slot=payload.slot,
        symbol=payload.symbol,
        depth=payload.depth,
        book_path_s3=s3_uri,
        meta=meta,
    )


def main():
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -m pipeline.data.ingest_orderbook '<json_payload>'",
            file=sys.stderr,
        )
        raise SystemExit(2)
    payload_raw = sys.argv[1]
    payload = IngestOrderbookInput.model_validate_json(payload_raw)
    out = run(payload)
    print(out.model_dump_json())


if __name__ == "__main__":
    main()
