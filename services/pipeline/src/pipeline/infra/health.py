"""Minimal HTTP health endpoint."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Tuple

from .healthcheck import run


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - required name
        status, body = self._health_response()
        self.send_response(status)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - inherited
        """Silence the default HTTP server logging."""
        return

    def _health_response(self) -> Tuple[int, bytes]:
        if self.path != "/health":
            return 404, b"Not Found"
        ok = run()
        return (200, b"OK") if ok else (500, b"FAIL")


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:  # nosec B104
    """Start health check HTTP server."""
    HTTPServer((host, port), _Handler).serve_forever()


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
