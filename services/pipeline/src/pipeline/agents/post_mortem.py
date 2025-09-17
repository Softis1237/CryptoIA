from __future__ import annotations

"""
PostMortemAgent — агент разбора полётов, запускается после закрытия сделки.

Вход: trade_context с ключами:
  - run_id: связка с релизом/постом;
  - symbol, side, entry_price, exit_price, qty, pnl;
  - chain: упрощённая цепочка причин (strategic_verdicts, дебаты, новости);

Выход: lesson (reason, context, recommendation) → agent_lessons.
При наличии LLM — используется call_openai_json; иначе — эвристическое резюме.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..infra.db import insert_agent_lesson, upsert_agent_lesson_embedding_by_hash
from .embeddings import embed_text
from ..reasoning.llm import call_openai_json


@dataclass
class PostMortemInput:
    trade: Dict[str, Any]
    scope: str = "trading"


def _fallback_lesson(tr: Dict[str, Any]) -> Dict[str, str]:
    side = str(tr.get("side") or "").upper()
    entry = float(tr.get("entry_price") or 0.0)
    exitp = float(tr.get("exit_price") or 0.0)
    pnl = float(tr.get("pnl") or (exitp - entry) * (1 if side == "LONG" else -1))
    ok = pnl >= 0
    reason = "followed_trend" if ok else "ignored_htf_trend"
    recommendation = (
        "поддерживать правила тренда и риск‑менеджмента"
        if ok
        else "снизить вес сигналов против HTF‑направления и увеличить ATR‑зазор"
    )
    return {
        "title": f"{side} итог: {'прибыль' if ok else 'убыток'}",
        "insight": f"Вход {entry:.2f} → выход {exitp:.2f}; pnl={pnl:.2f}",
        "action": recommendation,
        "risk": "проверять новости и ликвидность"
    }


def _compute_hash(lesson: Dict[str, str]) -> str:
    import hashlib, json
    norm = json.dumps(lesson, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def run(inp: PostMortemInput) -> Dict[str, Any]:
    tr = dict(inp.trade or {})
    sys = (
        "Ты аналитик риск‑комитета. Тебе дан контекст сделки и цепочка решений. "
        "Сформулируй короткий урок (JSON: {reason, context, recommendation})."
    )
    usr = str(tr)
    lesson_struct: Optional[Dict[str, str]] = None
    try:
        raw = call_openai_json(sys, usr)
        if raw and isinstance(raw, dict):
            lesson_struct = {
                "title": (raw.get("reason") or "Урок")[:64],
                "insight": str(raw.get("context") or "").strip(),
                "action": str(raw.get("recommendation") or "").strip(),
                "risk": str(raw.get("risk") or "").strip() or "",
            }
    except Exception:
        lesson_struct = None
    lesson = lesson_struct or _fallback_lesson(tr)
    try:
        # persist lesson with unique hash in meta for dedup and embedding upsert
        h = _compute_hash(lesson)
        meta = {"source": "postmortem", "run_id": tr.get("run_id"), "hash": h, "title": lesson.get("title"), "insight": lesson.get("insight"), "action": lesson.get("action"), "risk": lesson.get("risk")}
        insert_agent_lesson(lesson, scope=inp.scope, meta=meta)
        # embedding (best-effort)
        try:
            vec = embed_text(" | ".join([lesson.get("title", ""), lesson.get("insight", ""), lesson.get("action", "")]))
            upsert_agent_lesson_embedding_by_hash(inp.scope, h, vec)
        except Exception:
            pass
    except Exception:
        pass
    return {"status": "ok", "lesson": lesson}


def main() -> None:
    import sys, json
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.agents.post_mortem '<json_payload>'", file=sys.stderr)
        sys.exit(2)
    payload = PostMortemInput(**json.loads(sys.argv[1]))
    out = run(payload)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
