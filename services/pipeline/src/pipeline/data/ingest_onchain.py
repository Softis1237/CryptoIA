from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests  # type: ignore[import-untyped]
from loguru import logger
from pydantic import BaseModel

try:
    # Optional import to satisfy typing and future usage
    from dune_client.client import DuneClient  # type: ignore
except Exception:  # pragma: no cover - optional runtime dep
    DuneClient = object  # type: ignore

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


def _get_env_int(name: str) -> Optional[int]:
    val = os.getenv(name)
    if not val:
        return None
    try:
        return int(val)
    except Exception:  # noqa: BLE001
        logger.warning(f"Invalid int for {name}={val!r}; skipping")
        return None


def _parse_ts(value) -> int:
    if value is None:
        return int(datetime.now(timezone.utc).timestamp())
    # integer or float epoch (sec or ms)
    try:
        v = float(value)
        if v > 1e12:  # likely ms
            v = v / 1000.0
        return int(v)
    except Exception:  # noqa: BLE001
        pass
    # try parse as datetime string
    try:
        dt = pd.to_datetime(value, utc=True)
        if isinstance(dt, pd.Timestamp) and not pd.isna(dt):
            return int(dt.to_pydatetime().replace(tzinfo=timezone.utc).timestamp())
    except Exception:  # noqa: BLE001
        pass
    return int(datetime.now(timezone.utc).timestamp())


def _row_to_signal(metric: str, row: dict) -> OnchainSignal:
    # Extract timestamp
    ts_keys = [
        "ts",
        "time",
        "timestamp",
        "block_time",
        "date",
        "block_date",
    ]
    ts_val = None
    for k in ts_keys:
        if k in row and row[k] is not None:
            ts_val = row[k]
            break
    ts = _parse_ts(ts_val)

    # Extract numeric value from known keys first
    value_keys = [
        "value",
        "val",
        "count",
        "sum",
        "avg",
        "mvrv",
        "z_score",
        "netflow",
        "active_addresses",
        "sopr",
        "balance",
        "volume",
    ]
    value: Optional[float] = None
    for k in value_keys:
        if k in row and row[k] is not None:
            try:
                value = float(row[k])
                break
            except Exception:  # noqa: BLE001
                continue
    # Fallback: first numeric column not in ts_keys
    if value is None:
        for k, v in row.items():
            if k in ts_keys or v is None:
                continue
            try:
                value = float(v)
                break
            except Exception:  # noqa: BLE001
                continue

    return OnchainSignal(metric=metric, value=value, ts=ts)


def _fetch_from_dune(dune_client: DuneClient, query_id: int, api_key: str) -> dict:
    """Execute a Dune query and return the last row of the result.

    Uses Dune HTTP API directly for stability. Requires DUNE_API_KEY.
    """
    headers = {"x-dune-api-key": api_key}
    base = "https://api.dune.com/api/v1"
    # Try to read latest cached result (if exists)
    try:
        r = requests.get(
            f"{base}/query/{query_id}/results",
            params={"limit": 10000},
            headers=headers,
            timeout=30,
        )
        if r.status_code == 200:
            js = r.json() or {}
            rows = ((js.get("result") or {}).get("rows") or [])
            if rows:
                return rows[-1]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Dune cached results read failed for {query_id}: {exc}")

    # Execute query
    try:
        exec_resp = requests.post(
            f"{base}/query/{query_id}/execute",
            json={"parameters": []},
            headers=headers,
            timeout=30,
        )
        exec_resp.raise_for_status()
        execution_id = (exec_resp.json() or {}).get("execution_id")
        if not execution_id:
            raise RuntimeError("No execution_id in Dune response")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to execute Dune query {query_id}: {exc}")

    # Poll until completion
    for _ in range(20):  # ~40s max
        try:
            st = requests.get(
                f"{base}/execution/{execution_id}/status", headers=headers, timeout=15
            )
            st.raise_for_status()
            state = (st.json() or {}).get("state")
            if state == "QUERY_STATE_COMPLETED":
                break
            if state in {"QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"}:
                raise RuntimeError(f"Dune query {query_id} failed: state={state}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Dune status check failed for {query_id}: {exc}")
        time.sleep(2)
    else:
        raise TimeoutError(f"Dune query {query_id} timed out")

    # Fetch results
    res = requests.get(
        f"{base}/execution/{execution_id}/results",
        params={"limit": 10000},
        headers=headers,
        timeout=60,
    )
    res.raise_for_status()
    data = res.json() or {}
    rows = ((data.get("result") or {}).get("rows") or [])
    if not rows:
        raise RuntimeError(f"Dune query {query_id} returned no rows")
    return rows[-1]


def run(payload: IngestOnchainInput) -> IngestOnchainOutput:
    dune_api_key = os.getenv("DUNE_API_KEY")
    if not dune_api_key:
        logger.warning("DUNE_API_KEY is not set; skipping on-chain ingestion")
        return IngestOnchainOutput(
            run_id=payload.run_id,
            slot=payload.slot,
            asset=payload.asset,
            onchain_signals=[],
            onchain_path_s3="",
        )

    # Build metric -> Dune query id map from env
    # Recommend configuring queries to return columns: ts (int|timestamp) and value (numeric)
    METRIC_TO_QUERY_ID: dict[str, Optional[int]] = {
        "active_addresses": _get_env_int("DUNE_QUERY_ACTIVE_ADDRESSES"),
        "exchanges_netflow_sum": _get_env_int("DUNE_QUERY_EXCHANGES_NETFLOW_SUM"),
        "mvrv_z_score": _get_env_int("DUNE_QUERY_MVRV_Z_SCORE"),
        "sopr": _get_env_int("DUNE_QUERY_SOPR"),
        "miners_balance_sum": _get_env_int("DUNE_QUERY_MINERS_BALANCE_SUM"),
        "transfers_volume_sum": _get_env_int("DUNE_QUERY_TRANSFERS_VOLUME_SUM"),
    }

    # Initialize Dune client instance (kept for parity, though we use HTTP above)
    try:
        dune_client = DuneClient(dune_api_key)  # type: ignore[call-arg]
    except Exception:
        dune_client = None  # type: ignore[assignment]

    signals: List[OnchainSignal] = []
    for metric, qid in METRIC_TO_QUERY_ID.items():
        if not qid or qid <= 0:
            logger.warning(f"No Dune query id configured for {metric}; skipping")
            signals.append(
                OnchainSignal(
                    metric=metric,
                    value=None,
                    ts=int(datetime.now(timezone.utc).timestamp()),
                )
            )
            continue
        try:
            row = _fetch_from_dune(dune_client, qid, dune_api_key)
            sig = _row_to_signal(metric, row)
            signals.append(sig)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to fetch {metric} via Dune (query {qid}): {exc}")
            signals.append(
                OnchainSignal(
                    metric=metric,
                    value=None,
                    ts=int(datetime.now(timezone.utc).timestamp()),
                )
            )

    # Persist to S3 as parquet
    try:
        df = pd.DataFrame([s.model_dump() for s in signals])
        table = pa.Table.from_pandas(df)
        pq.write_table(table, "onchain.parquet", compression="zstd")
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        s3_path = f"runs/{date_key}/{payload.slot}/onchain.parquet"
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="zstd")
        buf = sink.getvalue().to_pybytes()
        s3_uri = upload_bytes(s3_path, buf, content_type="application/octet-stream")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to persist on-chain signals: {exc}")
        s3_uri = ""

    return IngestOnchainOutput(
        run_id=payload.run_id,
        slot=payload.slot,
        asset=payload.asset,
        onchain_signals=signals,
        onchain_path_s3=s3_uri,
    )
