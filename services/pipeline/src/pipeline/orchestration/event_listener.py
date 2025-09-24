from __future__ import annotations

"""Фоновый обработчик событий для master orchestrator."""

import argparse
import time
from typing import Sequence

from loguru import logger

from ..infra.db import fetch_pending_orchestrator_events
from .master_orchestrator_agent import MasterOrchestratorAgent


def process_once(agent: MasterOrchestratorAgent, slot: str) -> bool:
    pending: Sequence[dict] = fetch_pending_orchestrator_events(limit=10)
    if not pending:
        return False
    logger.info("event_listener: найдено %d событий, запускаем orchestrator", len(pending))
    agent.run({"slot": slot})
    return True


def loop(poll_interval: int, slot: str) -> None:
    agent = MasterOrchestratorAgent()
    while True:
        try:
            processed = process_once(agent, slot)
            if not processed:
                time.sleep(max(5, poll_interval))
        except KeyboardInterrupt:  # pragma: no cover
            logger.info("event_listener остановлен пользователем")
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("event_listener error: %s", exc)
            time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Background listener for orchestrator events")
    parser.add_argument("--poll", type=int, default=30, help="Интервал опроса в секундах")
    parser.add_argument("--slot", default="orchestrator", help="Слот orchestrator'а")
    args = parser.parse_args()
    loop(args.poll, args.slot)


if __name__ == "__main__":  # pragma: no cover
    main()

