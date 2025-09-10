from __future__ import annotations

import io
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests  # type: ignore[import-untyped]
from loguru import logger
from pydantic import BaseModel, Field

from ..infra.s3 import upload_bytes
from ..reasoning.llm import call_flowise_json
from ..reasoning.news_facts import extract_news_facts_batch


class NewsSignal(BaseModel):
    ts: int
    title: str
    url: str
    source: str
    sentiment: str
    impact_score: float
    topics: Optional[List[str]] = None


class IngestNewsInput(BaseModel):
    run_id: str
    slot: str
    time_window_hours: int = 12
    query: str = "crypto OR bitcoin"

    news_signals: List[dict] = Field(default_factory=list)
    news_facts: Optional[List[dict]] = None


class IngestNewsOutput(BaseModel):
    run_id: str
    slot: str
    news_signals: List[NewsSignal]
    news_facts: Optional[List[dict]] = None
    news_path_s3: str


def _get_news_sentiments_llm(
    news_items: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Process a batch of news items through a Flowise graph for sentiment, impact, and topics.
    Returns a dictionary mapping URL to its processed data.
    """
    if not news_items:
        return {}
    if os.getenv("ENABLE_NEWS_SENTIMENT", "0") not in {"1", "true", "True"}:
        logger.info("LLM-based news sentiment is disabled. Skipping.")
        return {}

    logger.info(f"Processing {len(news_items)} news items for sentiment with LLM...")
    try:
        # Prepare a compact representation for the prompt
        compact_news = [
            {"url": item["url"], "title": item["title"][:250]} for item in news_items
        ]

        system_prompt = (
            "You are a financial news analyst for the crypto market."
            " For each headline, provide a JSON object with: sentiment (positive|negative|neutral),"
            " impact_score (float 0.0-1.0 indicating market-moving potential), and topics (list of 1-3 relevant keywords)."
            " Respond ONLY with a JSON list of these objects, one for each headline."
        )
        user_prompt = f"Analyze these news headlines:\n{compact_news}"

        # Call the specific Flowise endpoint for sentiment analysis
        results = call_flowise_json(
            "FLOWISE_SENTIMENT_URL", {"system": system_prompt, "user": user_prompt}
        )

        if (
            not results
            or isinstance(results, dict)
            and results.get("status") == "error"
        ):
            logger.warning("Flowise sentiment call returned no valid data.")
            return {}
        if not isinstance(results, list):
            logger.warning("Flowise sentiment call returned unexpected format.")
            return {}

        # The result should be a list of dicts. We map it back to the original news items by URL.
        # It's assumed the LLM returns one object per input item in the same order.
        processed_data = {}
        if len(results) == len(news_items):
            for i, item in enumerate(news_items):
                processed_data[item["url"]] = results[i]
        else:
            logger.warning(
                f"Mismatch between input ({len(news_items)}) and output ({len(results)}) from sentiment LLM."
            )

        return processed_data

    except Exception as e:
        logger.error(f"Failed to process news sentiments with LLM: {e}")
        return {}


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
            logger.info("Fetching news from CryptoPanic...")
            rows = _fetch_from_cryptopanic(int(start.timestamp()), cryptopanic_key)
        elif newsapi_key:
            logger.info("Fetching news from NewsAPI...")
            rows = _fetch_from_newsapi(start, now, newsapi_key, inp.query)
        else:
            logger.warning("No news API key provided, cannot ingest news.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"fetch news failed: {e}")
        rows = []

    # Get enriched data from LLM
    llm_sentiments = _get_news_sentiments_llm(rows)

    signals: List[NewsSignal] = []
    for r in rows:
        url = r.get("url", "")
        # Get enriched data if available, otherwise fallback to a default
        enriched_data = llm_sentiments.get(url)

        if enriched_data:
            sentiment = enriched_data.get("sentiment", "neutral")
            impact_score = float(enriched_data.get("impact_score", 0.0))
            topics = enriched_data.get("topics", [])
        else:
            # Fallback to a simple heuristic if LLM fails or is disabled
            title = r.get("title", "").lower()
            if "bull" in title or "up" in title or "pump" in title:
                sentiment = "positive"
            elif (
                "bear" in title
                or "down" in title
                or "hack" in title
                or "exploit" in title
            ):
                sentiment = "negative"
            else:
                sentiment = "neutral"
            impact_score = 0.1  # low default impact
            topics = []

        ts_raw = r.get("ts")
        try:
            ts_int = int(pd.to_datetime(ts_raw, utc=True).timestamp())
        except Exception as e:
            logger.warning(f"Failed to parse news timestamp '{ts_raw}': {e}")
            ts_int = int(now.timestamp())

        signals.append(
            NewsSignal(
                ts=ts_int,
                title=r.get("title", ""),
                url=url,
                source=r.get("source", ""),
                sentiment=sentiment,
                impact_score=impact_score,
                topics=topics,
            )
        )

    logger.info(f"Generated {len(signals)} news signals.")

    # Fact extraction remains the same, it uses the generated signals
    logger.info("Extracting news facts...")
    news_facts = extract_news_facts_batch([s.model_dump() for s in signals]) or None
    logger.info(f"Extracted {len(news_facts) if news_facts else 0} facts.")

    df = pd.DataFrame([s.model_dump() for s in signals])
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    date_key = now.strftime("%Y-%m-%d")
    key = f"runs/{date_key}/{inp.slot}/news.parquet"
    news_path_s3 = upload_bytes(key, buf.getvalue(), "application/x-parquet")
    logger.info(f"Uploaded news signals to {news_path_s3}")
    return IngestNewsOutput(
        run_id=inp.run_id,
        slot=inp.slot,
        news_signals=signals,
        news_facts=news_facts,
        news_path_s3=news_path_s3,
    )
