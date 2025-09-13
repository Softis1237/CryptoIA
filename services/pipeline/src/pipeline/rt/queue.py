from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from loguru import logger


def _get_redis():
    try:
        import redis  # type: ignore

        host = os.getenv("REDIS_HOST", "redis")
        port = int(os.getenv("REDIS_PORT", "6379"))
        return redis.Redis(host=host, port=port, decode_responses=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Redis unavailable for RT queue: {e}")
        return None


QUEUE_KEY = os.getenv("RT_TRIGGER_QUEUE", "rt:triggers")


def publish_trigger(event: Dict[str, Any]) -> bool:
    """Publish trigger event to Redis list queue.

    Event shape: {type: str, ts: int, symbol: str, meta: dict}
    """
    r = _get_redis()
    if r is None:
        logger.warning("publish_trigger: redis unavailable; dropping event")
        return False
    try:
        payload = json.dumps(event, ensure_ascii=False)
        r.lpush(QUEUE_KEY, payload)
        # Also publish to pubsub channel for observers (best-effort)
        try:
            r.publish(QUEUE_KEY + ":pub", payload)
        except Exception:
            pass
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"publish_trigger failed: {e}")
        return False


def pop_trigger(timeout_s: int = 5) -> Optional[Dict[str, Any]]:
    r = _get_redis()
    if r is None:
        return None
    try:
        it = r.blpop(QUEUE_KEY, timeout=timeout_s)
        if not it:
            return None
        _, raw = it
        try:
            return json.loads(raw)
        except Exception:
            return None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"pop_trigger failed: {e}")
        return None


def get_key(key: str) -> Optional[str]:
    r = _get_redis()
    if r is None:
        return None
    try:
        return r.get(key)
    except Exception:
        return None


def setex(key: str, ttl: int, value: Any) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        r.setex(key, ttl, json.dumps(value) if not isinstance(value, str) else value)
    except Exception:
        pass


def incrby(key: str, amount: float) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        # store as float-like string
        cur = r.get(key)
        x = float(cur) if cur is not None else 0.0
        x += float(amount)
        r.set(key, str(x))
    except Exception:
        pass

