from __future__ import annotations

"""Minimal MCP-style server exposing a couple of safe tools over HTTP.

This is a lightweight scaffold (no external deps) for local usage. It accepts
POST /call with JSON {"tool": str, "params": dict}, and returns {"ok": bool, "result": any, "error": str|null}.

Tools implemented:
 - get_features_tail: params {features_s3: str, n: int=1, columns: list[str]|None}
 - levels_quantiles: params {features_s3: str, qs: list[float]=[0.2,0.5,0.8]}
 - news_top: params {news: list[dict], top_k: int=5}

This is not a full MCP spec implementation, but provides a clear seam to route
LLM tool use through a single controlled interface.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from ..infra.s3 import download_bytes


def _read_features(features_s3: str) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq
    raw = download_bytes(features_s3)
    table = pq.read_table(pa.BufferReader(raw))
    return table.to_pandas().sort_values("ts").reset_index(drop=True)


def tool_get_features_tail(params: Dict[str, Any]) -> Dict[str, Any]:
    s3 = str(params.get("features_s3"))
    n = int(params.get("n", 1))
    cols: Optional[List[str]] = params.get("columns")
    df = _read_features(s3)
    if cols is None:
        cols = [c for c in df.columns if c in {"ts", "close", "atr_14", "vwap"}]
    tail = df[cols].tail(max(1, n))
    return {"rows": tail.to_dict(orient="records"), "columns": cols}


def tool_levels_quantiles(params: Dict[str, Any]) -> Dict[str, Any]:
    s3 = str(params.get("features_s3"))
    qs = params.get("qs") or [0.2, 0.5, 0.8]
    df = _read_features(s3)
    close = df["close"].astype(float)
    levels = [float(close.quantile(float(q))) for q in qs]
    return {"levels": levels, "qs": [float(q) for q in qs]}


def tool_news_top(params: Dict[str, Any]) -> Dict[str, Any]:
    news = params.get("news") or []
    k = int(params.get("top_k", 5))
    try:
        sorted_news = sorted(news, key=lambda s: float(s.get("impact_score", 0.0) or 0.0), reverse=True)
    except Exception:
        sorted_news = news
    out = []
    for it in sorted_news[: max(1, k)]:
        out.append({
            "title": it.get("title"),
            "source": it.get("source"),
            "sentiment": it.get("sentiment"),
            "impact_score": float(it.get("impact_score", 0.0) or 0.0),
            "ts": it.get("ts"),
            "topics": it.get("topics"),
        })
    return {"news_top": out}


TOOLS = {
    "get_features_tail": tool_get_features_tail,
    "levels_quantiles": tool_levels_quantiles,
    "news_top": tool_news_top,
}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path != "/call":
            self._send({"ok": False, "error": "not_found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(data.decode("utf-8") or "{}")
            tool = str(payload.get("tool"))
            params = payload.get("params") or {}
            func = TOOLS.get(tool)
            if func is None:
                self._send({"ok": False, "error": f"unknown_tool:{tool}"}, status=400)
                return
            res = func(params)
            self._send({"ok": True, "result": res}, status=200)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"mcp server error: {e}")
            self._send({"ok": False, "error": str(e)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # suppress base HTTPServer logging; use loguru instead
        logger.info("mcp %s" % (format % args))

    def _send(self, obj: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(host: str = "0.0.0.0", port: int = 8765) -> None:
    httpd = HTTPServer((host, port), Handler)
    logger.info(f"MCP mini-server started on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def serve_in_thread(host: str = "127.0.0.1", port: int = 8765) -> threading.Thread:
    th = threading.Thread(target=serve, kwargs={"host": host, "port": port}, daemon=True)
    th.start()
    return th


def main():
    import os
    h = os.getenv("MCP_HOST", "0.0.0.0")
    p = int(os.getenv("MCP_PORT", "8765"))
    serve(h, p)


if __name__ == "__main__":
    main()

