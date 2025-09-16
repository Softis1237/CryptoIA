from __future__ import annotations

"""Minimal on-chain webhook receiver for Whale Watcher integration.

POST /onchain with JSON like:
{
  "symbol": "BTC/USDT",
  "exchange_netflow": -1500.0,  # negative => outflow from exchanges (bullish)
  "large_deposits": 1,
  "large_withdrawals": 3
}

Updates strategic_verdicts and whale_verdict_details accordingly.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from loguru import logger

from ..infra.health import start_background as start_health_server
from ..infra.db import upsert_strategic_verdict, upsert_whale_details


@dataclass
class WebConf:
    host: str = "0.0.0.0"
    port: int = 8787


def _handle_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(data.get("symbol") or "BTC/USDT")
    net = float(data.get("exchange_netflow", 0.0) or 0.0)
    dep = int(data.get("large_deposits", 0) or 0)
    wdr = int(data.get("large_withdrawals", 0) or 0)
    large_trades = dep + wdr
    now = datetime.now(timezone.utc)
    verdict = "WHALE_NEUTRAL"
    if net < 0 and wdr > dep:
        verdict = "WHALE_BULLISH"
    elif net > 0 and dep >= wdr:
        verdict = "WHALE_BEARISH"
    upsert_strategic_verdict("Whale Watcher", symbol, now, verdict, None, {"onchain": True, "netflow": net, "deposits": dep, "withdrawals": wdr})
    upsert_whale_details("Whale Watcher", symbol, now, float(net), int(large_trades), int(large_trades))
    return {"status": "ok", "verdict": verdict}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path != "/onchain":
            self._send({"ok": False, "error": "not_found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8") or "{}")
            res = _handle_payload(payload)
            self._send({"ok": True, "result": res}, status=200)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"onchain_webhook error: {e}")
            self._send({"ok": False, "error": str(e)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logger.info("onchain %s" % (format % args))

    def _send(self, obj: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(conf: WebConf) -> None:
    httpd = HTTPServer((conf.host, conf.port), Handler)
    logger.info(f"Onchain webhook on http://{conf.host}:{conf.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> None:
    start_health_server()
    serve(WebConf())


if __name__ == "__main__":
    main()

