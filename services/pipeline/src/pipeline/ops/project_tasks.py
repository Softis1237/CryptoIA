from __future__ import annotations

"""Интеграция с системой управления задачами для агентов."""

import json
import os
from typing import Iterable

from loguru import logger

try:  # pragma: no cover - опциональная зависимость
    import requests  # type: ignore
except Exception:  # noqa: BLE001
    requests = None  # type: ignore

from ..infra import db


def ensure_task(summary: str, description: str, priority: str = "medium", tags: Iterable[str] | None = None) -> None:
    """Создать задачу в БД и, при наличии вебхука, отправить уведомление."""

    tags_list = list(tags or [])
    task_id = db.insert_data_source_task(summary, description, priority, tags_list)
    webhook = os.getenv("PROJECT_TASKS_WEBHOOK")
    if not webhook or requests is None:
        return
    payload = {
        "summary": summary,
        "description": description,
        "priority": priority,
        "tags": tags_list,
        "task_id": task_id,
    }
    try:
        requests.post(webhook, data=json.dumps(payload), timeout=3)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_task: webhook failed: %s", exc)
