from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger


def _get_redis():
    try:
        import redis  # type: ignore

        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        return redis.Redis(host=host, port=port, decode_responses=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Redis unavailable: {e}")
        return None


def acquire_release_lock(slot: str, ttl_seconds: int = 1800, tz_name: str | None = None) -> tuple[bool, str]:
    tz = ZoneInfo(tz_name or os.getenv("TIMEZONE", "Asia/Jerusalem"))
    today = datetime.now(tz).strftime("%Y-%m-%d")
    key = f"release:{today}:{slot}"
    r = _get_redis()
    if not r:
        logger.warning("Redis lock skipped (redis unavailable)")
        return True, key
    try:
        ok = r.set(name=key, value="1", nx=True, ex=ttl_seconds)
        if not ok:
            logger.warning(f"Duplicate run prevented by lock: {key}")
            return False, key
        return True, key
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Redis lock error: {e}")
        return True, key


@contextmanager
def slot_lock(slot: str, ttl_seconds: int = 1800, tz_name: str | None = None):
    ok, key = acquire_release_lock(slot, ttl_seconds=ttl_seconds, tz_name=tz_name)
    if not ok:
        raise RuntimeError(f"slot lock already active for {key}")
    try:
        yield
    finally:
        # Locks истекают автоматически по TTL; явный release не требуется.
        pass
