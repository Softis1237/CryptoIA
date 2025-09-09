from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests  # type: ignore[import-untyped]
from pydantic import BaseModel

from ..infra.s3 import upload_bytes


class SocialSignal(BaseModel):
    metric: str
    value: float | None
    ts: int


class IngestSocialInput(BaseModel):
    run_id: str
    slot: str


class IngestSocialOutput(BaseModel):
    run_id: str
    slot: str
    social_signals: List[SocialSignal]
    social_path_s3: str


def run(payload: IngestSocialInput) -> IngestSocialOutput:
    url = "https://api.alternative.me/fng/?limit=1&format=json"
    signals: List[SocialSignal] = []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        item = (resp.json().get("data") or [{}])[0]
        value = item.get("value")
        ts = int(item.get("timestamp", datetime.now(timezone.utc).timestamp()))
        val = float(value) if value is not None else None
        signals.append(SocialSignal(metric="fear_greed", value=val, ts=ts))
    except Exception:
        signals.append(
            SocialSignal(metric="fear_greed", value=None, ts=int(datetime.now(timezone.utc).timestamp()))
        )

    df = pd.DataFrame([s.model_dump() for s in signals])
    table = pa.Table.from_pandas(df)
    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    s3_path = f"runs/{date_key}/{payload.slot}/social.parquet"
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    s3_uri = upload_bytes(s3_path, sink.getvalue().to_pybytes(), content_type="application/octet-stream")

    return IngestSocialOutput(
        run_id=payload.run_id,
        slot=payload.slot,
        social_signals=signals,
        social_path_s3=s3_uri,
    )
