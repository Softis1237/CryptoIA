from __future__ import annotations

"""Фоновый слушатель событий для MasterOrchestratorAgent."""

import os
import time

from loguru import logger

from ..infra.db import fetch_pending_orchestrator_events
from .master_orchestrator_agent import MasterOrchestratorAgent


def main() -> None:
    poll = float(os.getenv("ORCHESTRATOR_EVENT_POLL_SEC", "10"))
    agent = MasterOrchestratorAgent()
    logger.info("Orchestrator event listener started (poll=%.1fs)", poll)
    try:
        while True:
            pending = fetch_pending_orchestrator_events(limit=1)
            if pending:
                logger.info(
                    "Processing orchestrator events: %s",
                    ", ".join(str(evt.get("event_type")) for evt in pending),
                )
                agent.run({})
            time.sleep(poll)
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        logger.info("Event listener stopped")


if __name__ == "__main__":
    main()

