"""Order book ingestion utility.

Fetches order book data using ccxt, computes basic metrics and uploads
levels to S3 in Parquet format.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import boto3  # type: ignore[import]
import ccxt  # type: ignore[import]
import pandas as pd  # type: ignore[import]


@dataclass
class OrderBookMeta:
    """Summary information for an ingested order book."""

    exchange_id: str
    symbol: str
    timestamp: int
    metrics: Dict[str, float]
    s3_path: str


def _compute_metrics(
    bids: List[List[float]], asks: List[List[float]]
) -> Dict[str, float]:
    """Compute imbalance and depth ratio metrics."""

    bid_volume = sum(level[1] for level in bids)
    ask_volume = sum(level[1] for level in asks)
    top_bid = bids[0][0]
    top_ask = asks[0][0]

    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    depth_ratio = bid_volume / ask_volume if ask_volume else float("inf")
    spread = top_ask - top_bid
    mid_price = (top_ask + top_bid) / 2

    return {
        "imbalance": imbalance,
        "depth_ratio": depth_ratio,
        "spread": spread,
        "mid_price": mid_price,
    }


def ingest_orderbook(
    exchange_id: str,
    symbol: str,
    *,
    depth: int = 5,
    bucket: str,
    prefix: str,
) -> OrderBookMeta:
    """Fetch, persist and upload an order book.

    Args:
        exchange_id: ccxt exchange identifier.
        symbol: Market symbol (e.g. "BTC/USDT").
        depth: Number of levels per side to request.
        bucket: S3 bucket name.
        prefix: S3 key prefix for uploaded Parquet files.

    Returns:
        OrderBookMeta describing the ingested order book.
    """

    exchange_class: Any = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    order_book = exchange.fetch_order_book(symbol, limit=depth)

    bids: List[List[float]] = order_book["bids"]
    asks: List[List[float]] = order_book["asks"]

    rows = [
        {"side": "bid", "price": price, "amount": amount} for price, amount in bids
    ] + [{"side": "ask", "price": price, "amount": amount} for price, amount in asks]

    df = pd.DataFrame(rows)
    ts = order_book.get("timestamp") or int(datetime.utcnow().timestamp() * 1000)
    local_path = Path(f"/tmp/{exchange_id}_{symbol.replace('/', '')}_{ts}.parquet")
    df.to_parquet(local_path, index=False)

    s3_key = f"{prefix.rstrip('/')}/{local_path.name}"
    boto3.client("s3").upload_file(str(local_path), bucket, s3_key)

    metrics = _compute_metrics(bids, asks)

    return OrderBookMeta(
        exchange_id=exchange_id,
        symbol=symbol,
        timestamp=ts,
        metrics=metrics,
        s3_path=f"s3://{bucket}/{s3_key}",
    )


__all__ = ["ingest_orderbook", "OrderBookMeta"]
