from __future__ import annotations

import os
import re
from urllib.parse import urlparse
from loguru import logger

from ..reasoning.llm import call_flowise_json
from ..infra.db import insert_user_insight


_CREDIBLE_DOMAINS = {
    "www.coindesk.com",
    "cointelegraph.com",
    "www.blockworks.co",
    "bitcoinmagazine.com",
    "www.newsbtc.com",
    "decrypt.co",
    "www.theblock.co",
    "cryptoslate.com",
}


def _extract_url(text: str) -> str | None:
    m = re.search(r"https?://\S+", text)
    return m.group(0) if m else None


def _heuristic_scores(text: str) -> tuple[str, float, float]:
    t = (text or "").lower()
    url = _extract_url(text)
    truth = 0.5
    fresh = 0.7
    verdict = "uncertain"
    if url:
        try:
            d = urlparse(url).netloc.lower()
            if d in _CREDIBLE_DOMAINS or any(k in d for k in ("coindesk", "cointelegraph", "blockworks", "theblock")):
                truth = max(truth, 0.8)
        except Exception:
            pass
    if any(w in t for w in ["rumor", "слух", "scam", "fake"]):
        truth = min(truth, 0.4)
    if any(w in t for w in ["today", "сегодня", "счас", "hours ago", "час", "мин"]):
        fresh = max(fresh, 0.85)
    if re.search(r"\b2023\b", t):
        fresh = min(fresh, 0.3)
    if truth >= 0.75 and fresh >= 0.6:
        verdict = "true"
    elif truth <= 0.45:
        verdict = "false"
    return verdict, float(truth), float(fresh)


def _llm_scores(text: str) -> tuple[str, float, float] | None:
    try:
        res = call_flowise_json(
            "FLOWISE_VALIDATE_URL",
            {
                "system": (
                    "You are a fact-checker for crypto insights. Given a user text, "
                    "return JSON with fields: verdict(one of true,false,uncertain),"
                    " score_truth(0..1), score_freshness(0..1)."
                ),
                "user": text,
            },
        )
        if isinstance(res, dict):
            v = str(res.get("verdict") or "uncertain")
            st = float(res.get("score_truth") or 0.5)
            sf = float(res.get("score_freshness") or 0.7)
            return v, st, sf
    except Exception as e:  # noqa: BLE001
        logger.warning(f"LLM validation failed: {e}")
    return None


def evaluate_and_store_insight(user_id: int, text: str) -> dict:
    url = _extract_url(text)
    out = _llm_scores(text)
    if not out:
        out = _heuristic_scores(text)
    verdict, st, sf = out
    meta = {"source": "insight", "llm": bool(os.getenv("FLOWISE_VALIDATE_URL"))}
    ins_id = insert_user_insight(
        user_id=user_id,
        text=text.strip(),
        url=url,
        verdict=verdict,
        score_truth=st,
        score_freshness=sf,
        meta=meta,
    )
    return {"id": ins_id, "verdict": verdict, "score_truth": st, "score_freshness": sf}

