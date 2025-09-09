from __future__ import annotations

import json
import os
from typing import Any, Optional

import redis
from loguru import logger


def _redis() -> redis.Redis:
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    return redis.Redis(host=host, port=port, decode_responses=True)


def set_alias(key: str, s3_uri: str, ttl: int = 86400) -> None:
    r = _redis()
    r.setex(f"fs:alias:{key}", ttl, s3_uri)


def get_alias(key: str) -> Optional[str]:
    r = _redis()
    return r.get(f"fs:alias:{key}")


def set_json(key: str, obj: Any, ttl: int = 86400) -> None:
    r = _redis()
    r.setex(f"fs:json:{key}", ttl, json.dumps(obj))


def get_json(key: str) -> Optional[Any]:
    r = _redis()
    v = r.get(f"fs:json:{key}")
    return json.loads(v) if v else None

