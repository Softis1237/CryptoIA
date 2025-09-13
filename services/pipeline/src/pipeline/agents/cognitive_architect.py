from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..infra.db import fetch_recent_run_summaries, insert_agent_config
from ..reasoning.llm import call_openai_json


@dataclass
class CognitiveArchitectInput:
    analyze_n: int = 50
    target_agents: List[str] = None  # e.g., ["ChartReasoningAgent", "DebateArbiter"]


def run(inp: CognitiveArchitectInput) -> Dict[str, Any]:
    n = int(inp.analyze_n or 50)
    targets = inp.target_agents or ["ChartReasoningAgent", "DebateArbiter"]
    rows = fetch_recent_run_summaries(min(200, max(10, n)))
    if not rows:
        return {"status": "no-data"}
    sys = (
        "Ты AI-архитектор. По журналу недавних запусков найди системные слабости и предложи улучшенные промпты для агентов.\n"
        "Верни JSON {\"recommendations\":[{agent,system_prompt,parameters}],\"lessons\":[...]} где parameters — JSON с ключами model/temperature по необходимости."
    )
    usr = f"Recent summaries: {rows}\nTargets: {targets}"
    data = call_openai_json(sys, usr, model=None)
    recs = (data or {}).get("recommendations", []) or []
    out: List[Dict[str, Any]] = []
    for r in recs:
        try:
            agent = str(r.get("agent"))
            if agent not in targets:
                continue
            sp = r.get("system_prompt")
            params = r.get("parameters") or {}
            ver = insert_agent_config(agent, sp, params, make_active=True)
            out.append({"agent": agent, "version": ver})
        except Exception:
            continue
    return {"status": "ok", "updated": out, "lessons": (data or {}).get("lessons", [])}


def main():
    import sys, json
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.agents.cognitive_architect '<json_payload>'", file=sys.stderr)
        sys.exit(2)
    payload = CognitiveArchitectInput(**json.loads(sys.argv[1]))
    res = run(payload)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()

