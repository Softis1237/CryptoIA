from __future__ import annotations

from typing import List, Optional, Dict

from loguru import logger

from .llm import call_openai_json, call_flowise_json
from .schemas import DebateResponse


def debate(
    rationale_points: List[str],
    regime: Optional[str],
    news_top: List[str],
    neighbors: Optional[List[dict]] = None,
    memory: Optional[List[str]] = None,
    trust: Optional[Dict[str, float]] = None,
) -> tuple[str, List[str]]:
    sys = (
        "Ты арбитр: сведи аргументы в 3–6 пунктов (grounded-only).\n"
        "Верни JSON строго {\"bullets\":[string,...],\"risk_flags\":[string,...]} без лишних полей.\n"
        "Основывайся только на аргументах моделей, режиме, топ‑новостях и похожих окнах; не придумывай чисел.\n"
        "Если есть память прошлых релизов — учитывай её как контекст; если есть веса доверия к агентам — учитывай их при приоритизации аргументов."
    )
    usr = (
        f"Аргументы моделей: {rationale_points}\n"
        f"Режим: {regime}\n"
        f"Новости: {news_top}\n"
        f"Похожие окна: {neighbors}\n"
        f"Память: {memory or []}\n"
        f"Доверие: {trust or {}}"
    )
    raw = call_flowise_json("FLOWISE_DEBATE_URL", {"system": sys, "user": usr}) or call_openai_json(sys, usr)
    data = None
    try:
        if raw:
            data = DebateResponse.model_validate(raw)
    except Exception:
        data = None
    if not data:
        # Fallback: лаконично переформулируем rationale_points
        bullets = [p for p in rationale_points][:4] or ["Сигналы моделей нейтральны."]
        return "\n".join(f"• {b}" for b in bullets), []
    bullets = [str(b) for b in data.bullets][:6]
    flags = [str(r) for r in data.risk_flags][:6]
    return "\n".join(f"• {b}" for b in bullets), flags
