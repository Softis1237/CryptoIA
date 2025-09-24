from __future__ import annotations

"""Одноразовая проверка PROJECT_TASKS_WEBHOOK."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from types import SimpleNamespace

from pipeline.agents import strategic_data_agent as sda
from pipeline.agents.strategic import trust as trust_mod
from pipeline.ops import project_tasks

received: dict[str, str] = {}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(length).decode('utf-8')
        received['body'] = body
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args, **kwargs):  # noqa: D401, ANN001, ARG002
        return


def run_demo() -> dict[str, str | dict]:
    server = HTTPServer(('127.0.0.1', 8971), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    os.environ['PROJECT_TASKS_WEBHOOK'] = 'http://127.0.0.1:8971/hook'

    stub_db = SimpleNamespace(
        fetch_data_sources=lambda limit=200: [],
        upsert_data_source=lambda **_: None,
        insert_data_source_anomaly=lambda **_: None,
        insert_data_source_history=lambda **_: None,
        insert_data_source_task=lambda summary, description, priority, tags: 1,
        insert_orchestrator_event=lambda *args, **kwargs: None,
    )

    sda.db = stub_db  # type: ignore[assignment]
    trust_mod.db = stub_db  # type: ignore[assignment]
    project_tasks.db = stub_db  # type: ignore[assignment]

    agent = sda.StrategicDataAgent()
    result = agent.run({'run_id': 'webhook-test', 'keywords': ['bitcoin', 'liquidity']})

    server.shutdown()
    thread.join()
    return {'agent_output': result.output, 'webhook_payload': json.loads(received.get('body', '{}'))}


if __name__ == '__main__':
    report = run_demo()
    print(json.dumps(report, ensure_ascii=False, indent=2))
