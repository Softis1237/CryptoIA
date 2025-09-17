from __future__ import annotations

"""
Мини‑обёртка над HashiCorp Vault (или любым совместимым HTTP API) для безопасного чтения секретов.

Env:
  - VAULT_URL=https://vault.local:8200
  - VAULT_TOKEN=... (AppRole/JWT и пр.)
  - VAULT_PATH_PREFIX=secret/data/cryptoia

Если переменные не заданы — возвращаем из обычного окружения.
"""

import os
from typing import Optional

import json

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dep
    requests = None  # type: ignore


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    # prefer env first
    if name in os.environ:
        return os.getenv(name)
    url = os.getenv("VAULT_URL")
    token = os.getenv("VAULT_TOKEN")
    prefix = os.getenv("VAULT_PATH_PREFIX")
    if not url or not token or not prefix or not requests:
        return default
    path = f"{prefix}/{name.lower()}"
    try:
        resp = requests.get(
            f"{url}/v1/{path}",
            headers={"X-Vault-Token": token},
            timeout=5,
        )
        if resp.status_code != 200:
            return default
        data = resp.json()
        # KV v2 path
        val = (
            data.get("data", {}).get("data", {}).get("value")
            or data.get("data", {}).get("value")
        )
        return str(val) if val is not None else default
    except Exception:
        return default

