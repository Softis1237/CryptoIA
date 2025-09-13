from __future__ import annotations

import io
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests  # type: ignore[import-untyped]
from loguru import logger
from pydantic import BaseModel, Field

from ..infra.s3 import upload_bytes
from ..infra.metrics import push_values
from ..reasoning.llm import call_flowise_json
from ..reasoning.news_facts import extract_news_facts_batch
import asyncio
import feedparser  # type: ignore[import-untyped]

try:
    import aiohttp  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional import; requirements should include aiohttp
    aiohttp = None  # type: ignore[assignment]


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


def _parse_dt_safe(val: Any) -> Optional[datetime]:
    try:
        return pd.to_datetime(val, utc=True).to_pydatetime()
    except Exception:
        return None


def _rss_default_sources() -> List[str]:
    """Build the list of RSS sources.

    Priority:
    1) Built-in compact defaults (major crypto outlets)
    2) Optional additions from env `RSS_NEWS_SOURCES` (comma/space/newline sep)
    3) Optional file list from `RSS_SOURCES_FILE` (one URL per line)
       If not set, we also look for a bundled `rss_sources_full.txt` next to this module.
    """
    from pathlib import Path

    # Compact, high-signal default list; can be extended via env/file
    defaults: List[str] = [
        os.getenv(
            "COINDESK_RSS_URL",
            "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
        ),
        os.getenv("COINTELEGRAPH_RSS_URL", "https://cointelegraph.com/rss"),
        "https://bitcoinmagazine.com/.rss/full/",
        "https://cryptoslate.com/feed/",
        "https://www.blockworks.co/feed",
        "https://news.bitcoin.com/feed/",
        "https://decrypt.co/feed",
        "https://www.theblock.co/rss.xml",
        "https://ambcrypto.com/feed/",
        "https://www.newsbtc.com/feed/",
    ]

    # 2) Allow override/extra via RSS_NEWS_SOURCES (comma/newline/space separated)
    extra_raw = os.getenv("RSS_NEWS_SOURCES", "").strip()
    if extra_raw:
        parts = [p.strip() for p in extra_raw.replace("\n", ",").replace(" ", ",").split(",")]
        defaults.extend([p for p in parts if p])

    # 3) Optional file: explicit path via env, else bundled full list
    urls_from_file: List[str] = []
    file_hint = os.getenv("RSS_SOURCES_FILE", "").strip()
    candidate_files: List[Path] = []
    if file_hint:
        candidate_files.append(Path(file_hint))
    # bundled full list near this file
    bundled = Path(__file__).with_name("rss_sources_full.txt")
    if bundled.exists():
        candidate_files.append(bundled)
    for f in candidate_files:
        try:
            if f.exists():
                with f.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        urls_from_file.append(line)
        except Exception:
            # ignore file errors; proceed with defaults
            pass
    defaults.extend(urls_from_file)

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: List[str] = []
    for u in defaults:
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out


async def _fetch_one_rss(session: "aiohttp.ClientSession", url: str, timeout_sec: float) -> Tuple[str, Optional[str]]:
    try:
        async with session.get(url, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return url, None
            return url, await resp.text()
    except Exception:
        return url, None


async def _fetch_rss_async(sources: List[str]) -> Dict[str, Optional[str]]:
    # timeout per request
    timeout_sec = float(os.getenv("RSS_TIMEOUT_SEC", "10"))
    # concurrency cap
    limit = int(os.getenv("RSS_CONCURRENCY", "8"))
    if aiohttp is None:
        logger.warning("aiohttp not available, RSS ingestion disabled")
        return {}
    connector = aiohttp.TCPConnector(limit_per_host=limit)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_one_rss(session, u, timeout_sec) for u in sources]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return {u: body for (u, body) in results}


def _fetch_from_rss_window(start: datetime) -> Tuple[List[dict], Dict[str, float]]:
    sources = _rss_default_sources()
    if not sources:
        return [], {"rss_total": 0.0, "rss_ok": 0.0, "rss_fail": 0.0}
    logger.info(f"Fetching news via RSS from {len(sources)} sources (fallback)...")
    try:
        bodies = asyncio.run(_fetch_rss_async(sources))
    except RuntimeError:
        # In case we're already in an event loop (rare in this context)
        logger.warning("Async loop detected; creating nested loop for RSS fetch")
        bodies = asyncio.get_event_loop().run_until_complete(_fetch_rss_async(sources))  # type: ignore[call-arg]

    items: List[dict] = []
    seen_urls: set[str] = set()
    start_ts = int(start.timestamp())
    rss_total = float(len(sources))
    rss_ok = float(sum(1 for b in bodies.values() if b))
    rss_fail = float(rss_total - rss_ok)
    for src_url, body in bodies.items():
        if not body:
            continue
        parsed = feedparser.parse(body)
        feed_title = (parsed.feed.get("title") if getattr(parsed, "feed", None) else None) or src_url
        for e in parsed.entries or []:
            url = getattr(e, "link", None) or getattr(e, "id", None) or ""
            if not url or url in seen_urls:
                continue
            # determine best-effort published/updated time
            ts_dt: Optional[datetime] = None
            if getattr(e, "published", None):
                ts_dt = _parse_dt_safe(getattr(e, "published"))
            if ts_dt is None and getattr(e, "updated", None):
                ts_dt = _parse_dt_safe(getattr(e, "updated"))
            if ts_dt is None and getattr(e, "published_parsed", None):
                try:
                    ts_dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    ts_dt = None
            if ts_dt is None and getattr(e, "updated_parsed", None):
                try:
                    ts_dt = datetime(*e.updated_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    ts_dt = None
            if ts_dt is None:
                ts_dt = datetime.now(timezone.utc)
            ts_int = int(ts_dt.timestamp())
            if ts_int < start_ts:
                continue
            title = (getattr(e, "title", None) or "").strip()
            items.append({
                "ts": ts_int,
                "title": title,
                "url": url,
                "source": feed_title,
            })
            seen_urls.add(url)
    # Sort desc by ts for consistency
    items.sort(key=lambda x: x.get("ts", 0), reverse=True)
    logger.info(f"Fetched {len(items)} RSS news items after dedup.")
    return items, {"rss_total": rss_total, "rss_ok": rss_ok, "rss_fail": rss_fail}


def run(inp: IngestNewsInput) -> IngestNewsOutput:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=inp.time_window_hours)
    cryptopanic_key = os.getenv("CRYPTOPANIC_TOKEN") or os.getenv("CRYPTOPANIC_API_KEY")
    newsapi_key = os.getenv("NEWSAPI_KEY") or os.getenv("NEWSAPI_API_KEY")

    rows: List[dict] = []
    rss_stats: Dict[str, float] = {}
    try:
        if cryptopanic_key:
            logger.info("Fetching news from CryptoPanic...")
            rows = _fetch_from_cryptopanic(int(start.timestamp()), cryptopanic_key)
        elif newsapi_key:
            logger.info("Fetching news from NewsAPI...")
            rows = _fetch_from_newsapi(start, now, newsapi_key, inp.query)
        else:
            # Free fallback: RSS feeds
            rows, rss_stats = _fetch_from_rss_window(start)
            if not rows:
                logger.warning("No news API key and RSS fallback returned no items.")
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
    # Push basic observability values if configured
    try:
        values = {"news_count": float(len(signals))}
        # enrich with RSS fetch stats when available
        values.update(rss_stats)
        push_values(job="ingest_news", values=values, labels={"slot": inp.slot})
    except Exception:
        pass

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
