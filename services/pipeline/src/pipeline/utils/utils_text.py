from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import os


sia = SentimentIntensityAnalyzer()


@dataclass
class Sentiment:
    label: str
    score: float


def compute_sentiment(text: str) -> Sentiment:
    # VADER is strong on short headlines
    s_vader = sia.polarity_scores(text)["compound"]
    # TextBlob polarity [-1,1]
    s_blob = TextBlob(text).sentiment.polarity
    # Blend with higher weight to VADER
    score = 0.65 * s_vader + 0.35 * s_blob
    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "neutral"
    return Sentiment(label=label, score=float(score))


_HF_PIPE = None


def _hf_sentiment(text: str) -> Sentiment | None:
    global _HF_PIPE
    try:
        if os.getenv("USE_HF_SENTIMENT", "0") not in {"1", "true", "True"}:
            return None
        if _HF_PIPE is None:
            from transformers import pipeline
            # Use a multilingual star-rating model by default for headlines
            model_name = os.getenv("HF_SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")
            _HF_PIPE = pipeline("sentiment-analysis", model=model_name)
        res = _HF_PIPE(text[:512])  # type: ignore[misc]
        if not res:
            return None
        out = res[0]
        label_raw = str(out.get("label", "")).lower()
        score = float(out.get("score", 0.5))
        if "neg" in label_raw or label_raw.startswith("1"):
            label = "negative"
            s = -score
        elif "pos" in label_raw or label_raw.startswith("5"):
            label = "positive"
            s = score
        else:
            label = "neutral"
            s = 0.0
        return Sentiment(label=label, score=float(s))
    except Exception:
        return None


def compute_sentiment_smart(text: str) -> Sentiment:
    """Sentiment with optional HF model, fallback to VADER+TextBlob.

    Returns Sentiment(label in {positive, neutral, negative}, score in [-1,1]).
    """
    s_hf = _hf_sentiment(text)
    if s_hf is not None:
        return s_hf
    return compute_sentiment(text)


TOPIC_KEYWORDS = {
    "etf": ["etf", "spot etf", "inflow", "outflow"],
    "regulation": ["sec", "regulator", "regulation", "law", "ban"],
    "exchange": ["binance", "coinbase", "bybit", "okx", "kraken", "exchange"],
    "hack": ["hack", "exploit", "vulnerability", "hackers", "bridge"],
    "macro": ["cpi", "inflation", "fed", "rate", "jobs", "macro"],
    "onchain": ["on-chain", "onchain", "whale", "addresses", "netflow"],
}


def extract_topics(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9+-]+", text.lower())
    topics: List[str] = []
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(k in tokens or k in text.lower() for k in kws):
            topics.append(topic)
    return topics[:4]
