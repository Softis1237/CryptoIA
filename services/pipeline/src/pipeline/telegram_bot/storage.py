"""Хранение пользовательских настроек в Redis."""

from __future__ import annotations

from typing import Dict

from loguru import logger

from ..infra.config import settings

try:
    from redis import Redis  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    Redis = None  # type: ignore


def _get_redis() -> Redis | None:
    if Redis is None:
        return None
    try:
        return Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    except Exception as e:  # pragma: no cover - network errors
        logger.warning(f"Redis unavailable: {e}")
        return None


def load_user_settings(user_id: int) -> Dict[str, str]:
    """Возвращает настройки пользователя или значения по умолчанию."""
    r = _get_redis()
    if not r:
        return {"lang": "ru", "freq": "daily"}
    try:
        data = r.hgetall(f"tg:user:{user_id}")
    except Exception as e:  # pragma: no cover - network errors
        logger.warning(f"Redis error: {e}")
        return {"lang": "ru", "freq": "daily"}
    return {"lang": data.get("lang", "ru"), "freq": data.get("freq", "daily")}


def save_user_setting(user_id: int, key: str, value: str) -> None:
    r = _get_redis()
    if not r:
        return
    try:
        r.hset(f"tg:user:{user_id}", key, value)
    except Exception as e:  # pragma: no cover - network errors
        logger.warning(f"Redis error: {e}")
