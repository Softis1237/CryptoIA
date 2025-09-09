from __future__ import annotations

import io
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests  # type: ignore[import-untyped]
from loguru import logger
from pydantic import BaseModel
from textblob import TextBlob

from ..infra.s3 import upload_bytes
from ..reasoning.news_facts import extract_news_facts_batch


class NewsSignal(BaseModel):
    ts: int
    title: str
    url: str
    source: str
    sentiment: str
    impact_score: float


class IngestNewsInput(BaseModel):
    run_id: str
    slot: str
    time_window_hours: int = 12
    query: str = "crypto OR bitcoin"


class IngestNewsOutput(BaseModel):
    run_id: str
    slot: str
    news_signals: List[NewsSignal]
    news_facts: Optional[List[dict]] = None
    news_path_s3: str


def _sentiment(text: str) -> tuple[str, float]:
    blob = TextBlob(text or "")
    score = float(blob.sentiment.polarity)
    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, abs(score)


def _fetch_from_cryptopanic(since_ts: int, api_key: str) -> List[dict]:
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": api_key,
        "public": "true",
        "kind": "news",
        "filter": "rising",
        "since": since_ts,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("results", [])
    items = []
    for item in data:
        items.append(
            {
                "ts": item.get("published_at"),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("source", {}).get("domain", ""),
            }
        )
    return items


def _fetch_from_newsapi(
    start: datetime, end: datetime, api_key: str, query: str
) -> List[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": api_key,
        "q": query,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "language": "en",
        "pageSize": 100,
        "sortBy": "publishedAt",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("articles", [])
    items = []
    for item in data:
        items.append(
            {
                "ts": item.get("publishedAt"),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("source", {}).get("name", ""),
            }
        )
    return items


def run(inp: IngestNewsInput) -> IngestNewsOutput:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=inp.time_window_hours)
    cryptopanic_key = os.getenv("CRYPTOPANIC_TOKEN") or os.getenv("CRYPTOPANIC_API_KEY")
    newsapi_key = os.getenv("NEWSAPI_KEY") or os.getenv("NEWSAPI_API_KEY")

    rows: List[dict] = []
    try:
        if cryptopanic_key:
            rows = _fetch_from_cryptopanic(int(start.timestamp()), cryptopanic_key)
        elif newsapi_key:
            rows = _fetch_from_newsapi(start, now, newsapi_key, inp.query)
        else:
            logger.warning("No news API key provided")
    except Exception as e:  # noqa: BLE001
        logger.error(f"fetch news failed: {e}")
        rows = []

    signals: List[NewsSignal] = []
    for r in rows:
        sent, impact = _sentiment(r.get("title", ""))
        ts_raw = r.get("ts")
        try:
            ts_int = int(pd.to_datetime(ts_raw, utc=True).timestamp())
        except Exception:
            ts_int = int(now.timestamp())
        signals.append(
            NewsSignal(
                ts=ts_int,
                title=r.get("title", ""),
                url=r.get("url", ""),
                source=r.get("source", ""),
                sentiment=sent,
                impact_score=impact,
            )
        )

    news_facts = extract_news_facts_batch([s.model_dump() for s in signals]) or None

    df = pd.DataFrame([s.model_dump() for s in signals])
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    date_key = now.strftime("%Y-%m-%d")
    key = f"runs/{date_key}/{inp.slot}/news.parquet"
    news_path_s3 = upload_bytes(key, buf.getvalue(), "application/x-parquet")
    return IngestNewsOutput(
        run_id=inp.run_id,
        slot=inp.slot,
        news_signals=signals,
        news_facts=news_facts,
        news_path_s3=news_path_s3,
    )
