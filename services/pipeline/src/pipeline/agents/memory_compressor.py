from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from ..mcp.client import call_tool as _call_tool
from ..infra.db import insert_agent_lesson
from ..reasoning.llm import call_openai_json


@dataclass
class MemoryCompressInput:
    n: int = 20  # take last N run summaries
    scope: str = "global"


def _lessons_to_unique_payloads(raw_lessons: Iterable[Any]) -> list[dict[str, str]]:
    """Normalize LLM память в компактные карточки и убираем дубликаты."""
    uniq: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for item in raw_lessons:
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "Lesson")
            insight = str(item.get("insight") or item.get("summary") or item.get("lesson") or "")
            action = str(item.get("action") or item.get("recommendation") or "")
            risk = str(item.get("risk") or item.get("risk_flags") or item.get("context") or "")
        else:
            text = str(item or "").strip()
            if not text:
                continue
            title = text[:80]
            insight = text
            action = ""
            risk = ""

        payload = {
            "title": title.strip() or "Lesson",
            "insight": insight.strip(),
            "action": action.strip(),
            "risk": risk.strip(),
        }
        norm = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        lesson_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if lesson_hash in seen_hashes:
            continue
        seen_hashes.add(lesson_hash)
        uniq.append(payload)
        if len(uniq) >= 7:
            break
    return uniq


def run(inp: MemoryCompressInput) -> Dict[str, Any]:
    # Pull recent summaries via MCP tool to reuse existing code path
    items = _call_tool("get_recent_run_summaries", {"n": int(inp.n)}) or {"items": []}
    rows = items.get("items") or []
    if not rows:
        return {"status": "no-data", "inserted": 0, "lessons": []}

    sys = (
        "Ты работаешь в риск-отделе трейдингового фонда. Тебе дают заметки о прошлых запусках системы. "
        "Сожми знания в 3-7 чётких уроков. Каждый урок — объект с ключами title, insight, action, risk. "
        "title: до 12 слов, конкретная тема. insight: что произошло и почему это важно. "
        "action: практический шаг на будущее. risk: ограничение или условие применения. "
        "Верни строго JSON {\"lessons\":[{...}]} без дополнительных полей. Не придумывай точные числа и не повторяйся."
    )
    usr = "Summaries: " + json.dumps(rows, ensure_ascii=False)
    data = call_openai_json(sys, usr)
    lessons = _lessons_to_unique_payloads(((data or {}).get("lessons") or []))
    inserted = 0
    stored: list[dict[str, Any]] = []
    for ls in lessons:
        try:
            insert_agent_lesson(
                json.dumps(ls, ensure_ascii=False),
                scope=inp.scope,
                meta={
                    "source": "compressor",
                    "n": inp.n,
                    "hash": hashlib.sha256(json.dumps(ls, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
                },
            )
            inserted += 1
            stored.append(ls)
        except Exception:
            continue
    return {"status": "ok", "inserted": inserted, "lessons": stored}


def main():
    import sys, json
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.agents.memory_compressor '<json_payload>'", file=sys.stderr)
        sys.exit(2)
    payload = MemoryCompressInput(**json.loads(sys.argv[1]))
    res = run(payload)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
