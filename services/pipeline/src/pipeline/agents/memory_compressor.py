from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..mcp.client import call_tool as _call_tool
from ..infra.db import insert_agent_lesson
from ..reasoning.llm import call_openai_json


@dataclass
class MemoryCompressInput:
    n: int = 20  # take last N run summaries
    scope: str = "global"


def run(inp: MemoryCompressInput) -> Dict[str, Any]:
    # Pull recent summaries via MCP tool to reuse existing code path
    items = _call_tool("get_recent_run_summaries", {"n": int(inp.n)}) or {"items": []}
    rows = items.get("items") or []
    if not rows:
        return {"status": "no-data", "inserted": 0, "lessons": []}

    sys = (
        "Ты ассистент-редактор. Сожми список заметок о прошлых запусках в 3-7 коротких практических уроков для трейдинга. "
        "Верни строго JSON {\"lessons\":[string,...]} без лишних полей. Избегай воды, дублирования и конкретных численных прогнозов."
    )
    usr = f"Summaries: {rows}"
    data = call_openai_json(sys, usr)
    lessons = [str(x) for x in ((data or {}).get("lessons") or [])][:7]
    inserted = 0
    for ls in lessons:
        try:
            insert_agent_lesson(ls, scope=inp.scope, meta={"source": "compressor", "n": inp.n})
            inserted += 1
        except Exception:
            continue
    return {"status": "ok", "inserted": inserted, "lessons": lessons}


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

