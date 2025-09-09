from __future__ import annotations

"""Tiny MCP client to call local MCP mini-server tools.

Reads base URL from MCP_URL or MCP_HOST/MCP_PORT. Returns dict or None on error.
"""

import os
from typing import Any, Dict, Optional

from loguru import logger


def _base_url() -> Optional[str]:
    url = os.getenv("MCP_URL")
    if url:
        return url.rstrip("/")
    host = os.getenv("MCP_HOST")
    port = os.getenv("MCP_PORT")
    if host and port:
        return f"http://{host}:{port}"
    return None


def call_tool(tool: str, params: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    url = _base_url()
    if not url:
        return None
    try:
        import requests

        r = requests.post(f"{url}/call", json={"tool": tool, "params": params}, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json() or {}
        if not data.get("ok"):
            return None
        return data.get("result")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"mcp client: call_tool failed for {tool}: {e}")
        return None

