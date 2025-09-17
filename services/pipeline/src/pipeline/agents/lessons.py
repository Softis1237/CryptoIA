from __future__ import annotations

"""
Релевантные уроки (Feedback Loop): отбор из agent_lessons для текущего контекста.

MVP: выбираем последние N уроков с непустыми полями title/insight/action. Если потребуется —
добавим векторный поиск (pgvector) и эмбеддинги через OpenAI/локальные модели.
"""

from typing import Any, Dict, List

from ..infra.db import fetch_recent_agent_lessons, fetch_agent_lessons_similar
from .embeddings import embed_text


def get_relevant_lessons(context: Dict[str, Any], k: int = 5) -> List[Dict[str, str]]:
    """Вернуть до k компактных уроков.

    Параметр context может включать regime, volatility, ta, news_points и т.п.
    Пока используем простой fallback: берём последние уроки и нормализуем поля.
    """
    # Try vector search first (if embeddings available)
    try:
        ctx_text = " | ".join(
            [
                str(context.get("regime") or ""),
                str(context.get("volatility") or ""),
                ", ".join(context.get("news") or []),
                str((context.get("ta") or {}).get("atr") or ""),
            ]
        ).strip()
        if ctx_text:
            vec = embed_text(ctx_text)
            rows = fetch_agent_lessons_similar(vec, k=k, scope=str(context.get("scope") or "global"))
        else:
            rows = []
    except Exception:
        rows = []
    if not rows:
        rows = fetch_recent_agent_lessons(k * 3)
    out: List[Dict[str, str]] = []
    for r in rows:
        try:
            txt = r.get("lesson_text") or ""
            meta = r.get("meta") or {}
            # structured duplication might exist — фильтруем по хэшу, если есть
            title = (meta.get("title") or "").strip() or None
            if not title and txt:
                title = txt[:64]
            if not title:
                continue
            action = (meta.get("action") or "").strip()
            insight = (meta.get("insight") or "").strip() or txt[:200]
            risk = (meta.get("risk") or "").strip()
            out.append({"title": title, "insight": insight, "action": action, "risk": risk})
            if len(out) >= k:
                break
        except Exception:
            continue
    return out
