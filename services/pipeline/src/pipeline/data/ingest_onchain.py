from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests  # type: ignore[import-untyped]
from loguru import logger
from pydantic import BaseModel

from ..infra.s3 import upload_bytes


class OnchainSignal(BaseModel):
    metric: str
    value: float | None
    ts: int


class IngestOnchainInput(BaseModel):
    run_id: str
    slot: str
    asset: str


class IngestOnchainOutput(BaseModel):
    run_id: str
    slot: str
    asset: str
    onchain_signals: List[OnchainSignal]
    onchain_path_s3: str


_METRIC_MAP = {
    "active_addresses": "addresses/active_count",
    "exchanges_netflow_sum": "exchanges/netflow_sum",
    "mvrv_z_score": "indicators/mvrv_z_score",
    "sopr": "indicators/sopr",
    "miners_balance_sum": "miners/balance_sum",
    "transfers_volume_sum": "transactions/transfers_volume_sum",
}


def _fetch_metric(
    metric: str, endpoint: str, asset: str, api_key: str
) -> OnchainSignal:
    url = f"https://api.glassnode.com/v1/metrics/{endpoint}"
    params = {"a": asset, "api_key": api_key, "i": "24h"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        last = data[-1]
    elif isinstance(data, dict):
        last = data
    else:
        last = {"t": int(datetime.now(timezone.utc).timestamp()), "v": None}
    ts = int(last.get("t") or last.get("time") or 0)
    value_raw = last.get("v")
    value = float(value_raw) if value_raw is not None else None
    return OnchainSignal(metric=metric, value=value, ts=ts)


def run(payload: IngestOnchainInput) -> IngestOnchainOutput:
    api_key = os.getenv("GLASSNODE_API_KEY")
    if not api_key:
        raise RuntimeError("GLASSNODE_API_KEY is not set")

    signals: List[OnchainSignal] = []
    for metric, endpoint in _METRIC_MAP.items():
        try:
            sig = _fetch_metric(metric, endpoint, payload.asset, api_key)
            signals.append(sig)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to fetch {metric}: {exc}")
            signals.append(
                OnchainSignal(
                    metric=metric,
                    value=None,
                    ts=int(datetime.now(timezone.utc).timestamp()),
                )
            )

    df = pd.DataFrame([s.model_dump() for s in signals])
    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    s3_path = f"runs/{date_key}/{payload.slot}/onchain.parquet"
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    buf = sink.getvalue().to_pybytes()
    s3_uri = upload_bytes(s3_path, buf, content_type="application/octet-stream")

    return IngestOnchainOutput(
        run_id=payload.run_id,
        slot=payload.slot,
        asset=payload.asset,
        onchain_signals=signals,
        onchain_path_s3=s3_uri,
    )
