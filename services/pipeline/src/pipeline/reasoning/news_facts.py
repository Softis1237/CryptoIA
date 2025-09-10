from __future__ import annotations

import os
from typing import Any, Dict, List

from loguru import logger

from .llm import call_flowise_json, call_openai_json


def extract_news_facts_batch(
    news: List[Dict[str, Any]], top_k: int | None = None
) -> List[Dict[str, Any]]:
    """Extract structured facts from news items via OpenAI JSON mode.

    Input items: {id, ts, title, source, sentiment?, impact_score?}
    Returns list of facts dicts: {src_id, ts, type, direction, magnitude, confidence, entities}
    Safe no-op: returns [] if disabled, no key, or budget exceeded.
    """
    try:
        if not news:
            return []
        if os.getenv("ENABLE_NEWS_FACTS", "0") not in {"1", "true", "True"}:
            return []
        items = sorted(
            news, key=lambda x: float(x.get("impact_score", 0.0) or 0.0), reverse=True
        )
        if top_k is None:
            top_k = int(os.getenv("NEWS_FACTS_TOPK", "10"))
        items = items[: max(1, top_k)]
        # Keep prompt compact
        compact = [
            {
                "id": str(x.get("id")),
                "ts": str(x.get("ts")),
                "title": (str(x.get("title")) or "")[:240],
                "source": str(x.get("source", ""))[:40],
            }
            for x in items
        ]
        system = (
            "You extract crypto market events from headlines."
            " Respond ONLY with JSON matching the schema."
        )
        user = (
            "Given top news headlines, extract significant factual events relevant to BTC/crypto.\n"
            "For each event output: src_id (from input id), ts (from input), type (one of: SEC_ACTION, ETF_APPROVAL, HACK, FORK, LISTING, MACRO, RUMOR_RETRACTION, OTHER),\n"
            "direction (bull|bear|volatility|neutral), magnitude [0..1], confidence [0..1], entities (list of strings).\n"
            "Skip vague or non-crypto events. If multiple events in one headline, split into separate items.\n"
            f"Input: {compact}"
        )
        data = call_flowise_json("FLOWISE_NEWS_URL", {"system": system, "user": user})
        if not data or data.get("status") == "error":
            data = call_openai_json(
                system_prompt=system,
                user_prompt=user,
                model=os.getenv("OPENAI_MODEL_NEWS_FACTS"),
            )
        if not isinstance(data, dict) or data.get("status") == "error":
            return []
        facts = data.get("facts") or data.get("events") or []
        if not isinstance(facts, list):
            if isinstance(data, list):
                facts = data
            else:
                return []
        # normalize
        allowed_types = {
            "SEC_ACTION",
            "ETF_APPROVAL",
            "HACK",
            "FORK",
            "LISTING",
            "MACRO",
            "RUMOR_RETRACTION",
            "OTHER",
        }
        allowed_dir = {"bull", "bear", "volatility", "neutral"}
        out: List[Dict[str, Any]] = []
        for f in facts:
            try:
                src_id = str(f.get("src_id") or f.get("id") or "")
                ts = str(f.get("ts") or "")
                typ = str(f.get("type") or "OTHER").upper()
                if typ not in allowed_types:
                    typ = "OTHER"
                direction = str(f.get("direction") or "neutral").lower()
                if direction not in allowed_dir:
                    direction = "neutral"
                mag = float(f.get("magnitude", 0.0) or 0.0)
                conf = float(f.get("confidence", 0.0) or 0.0)
                mag = 0.0 if mag < 0 else (1.0 if mag > 1 else mag)
                conf = 0.0 if conf < 0 else (1.0 if conf > 1 else conf)
                entities = f.get("entities") or []
                if not isinstance(entities, list):
                    entities = [str(entities)]
                out.append(
                    {
                        "src_id": src_id,
                        "ts": ts,
                        "type": typ,
                        "direction": direction,
                        "magnitude": mag,
                        "confidence": conf,
                        "entities": entities,
                    }
                )
            except Exception:
                continue
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning(f"extract_news_facts_batch failed: {e}")
        return []
