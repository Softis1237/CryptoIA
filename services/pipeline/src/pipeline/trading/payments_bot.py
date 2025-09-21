
"""Entry point for the Telegram payments/signals bot."""
from __future__ import annotations

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from loguru import logger

from pipeline.telegram_bot import main as telegram_main


def _start_health_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # type: ignore[override]
            if self.path == "/healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *_args, **_kwargs):  # noqa: D401
            pass

    server = HTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def main() -> None:
    _start_health_server()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not provided. Running in idle mode.")
        while True:
            time.sleep(60)
    telegram_main()


if __name__ == "__main__":
    main()
