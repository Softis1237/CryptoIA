from __future__ import annotations

import hashlib
import json as _json
import os
import time
from typing import Optional

from loguru import logger

from ..infra.metrics import push_values


def call_openai_json(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> Optional[dict]:
    """Call OpenAI Chat Completions expecting a JSON object response.

    Returns ``None`` if ``OPENAI_API_KEY`` is not set or on failure.
    """

    global _LLM_CALLS, _LLM_CACHE_HITS, _LLM_FAILURES

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — LLM call skipped (fallback)")
        return None

    if _rate_limited():
        logger.warning("LLM budget exceeded — skipping OpenAI call")
        return None

    try:
        from openai import OpenAI  # type: ignore[import-untyped]

        client = OpenAI(api_key=api_key)
        model_name: str = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

        ttl = int(os.getenv("LLM_CACHE_TTL_SEC", "600"))
        key = _hash_key("openai", model_name, system_prompt, user_prompt)
        cached = _try_cache_get(key)
        if cached is not None:
            _LLM_CACHE_HITS += 1
            _push_metrics()
            return cached

        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "400")),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content or "{}"
        data = _json.loads(text)
        _LLM_CALLS += 1
        _try_cache_put(key, data, ttl)
        _push_metrics()
        return data
    except Exception as e:  # noqa: BLE001
        logger.exception(f"OpenAI call failed: {e}")
        _LLM_FAILURES += 1
        _push_metrics()
        return None


def call_flowise_json(url_env: str, payload: dict) -> Optional[dict]:
    """Call Flowise endpoint specified by ``url_env`` and return JSON.

    Returns ``None`` if the endpoint is not configured or on error.
    """

    import json

    import requests  # type: ignore[import-untyped]

    global _LLM_CALLS, _LLM_CACHE_HITS, _LLM_FAILURES

    url = os.getenv(url_env)
    if not url:
        return None

    if _rate_limited():
        logger.warning("LLM budget exceeded — skipping Flowise call")
        return None

    timeout = float(os.getenv("FLOWISE_TIMEOUT_SEC", "15"))
    max_retries = int(os.getenv("FLOWISE_MAX_RETRIES", "2"))
    backoff = float(os.getenv("FLOWISE_BACKOFF_SEC", "1.0"))

    ttl = int(os.getenv("LLM_CACHE_TTL_SEC", "600"))
    key = _hash_key("flowise", url_env, _json.dumps(payload, ensure_ascii=False))
    cached = _try_cache_get(key)
    if cached is not None:
        _LLM_CACHE_HITS += 1
        _push_metrics()
        return cached

    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            if r.headers.get("content-type", "").startswith("application/json"):
                data = r.json()
            else:
                try:
                    data = json.loads(r.text)
                except Exception:
                    data = {"text": r.text}
            _LLM_CALLS += 1
            _try_cache_put(key, data, ttl)
            _push_metrics()
            return data
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries:
                sleep_for = backoff * (2**attempt)
                logger.warning(
                    f"Flowise call failed ({url_env}) attempt {attempt+1}/{max_retries+1}: {e}; retry in {sleep_for:.1f}s",
                )
                time.sleep(sleep_for)
                continue
            logger.warning(f"Flowise call failed ({url_env}) giving up: {e}")
            _LLM_FAILURES += 1
            _push_metrics()
            return None
    return None


# Simple in-process budget and counters
_LLM_CALLS = 0
_LLM_CACHE_HITS = 0
_LLM_FAILURES = 0


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _try_cache_get(key: str) -> Optional[dict]:
    if os.getenv("ENABLE_LLM_CACHE", "1") not in {"1", "true", "True"}:
        return None
    try:
        import redis  # type: ignore[import-untyped]

        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=0,
        )
        raw = r.get(f"llm:{key}")
        if not raw:
            return None
        data = _json.loads(raw)
        return data
    except Exception:
        return None


def _try_cache_put(key: str, data: dict, ttl: int) -> None:
    if os.getenv("ENABLE_LLM_CACHE", "1") not in {"1", "true", "True"}:
        return
    try:
        import redis  # type: ignore[import-untyped]

        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=0,
        )
        r.setex(f"llm:{key}", ttl, _json.dumps(data))
    except Exception:
        pass


def _rate_limited() -> bool:
    budget = int(os.getenv("LLM_CALLS_BUDGET", "8"))
    return _LLM_CALLS >= budget


def _push_metrics() -> None:
    try:
        push_values(
            job="llm",
            values={
                "llm_calls": float(_LLM_CALLS),
                "llm_cache_hits": float(_LLM_CACHE_HITS),
                "llm_failures": float(_LLM_FAILURES),
            },
            labels={},
        )
    except Exception:
        pass
