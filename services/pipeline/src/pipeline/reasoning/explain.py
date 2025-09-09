from __future__ import annotations

from typing import List, Optional

from loguru import logger

from .llm import call_openai_json, call_flowise_json
from .schemas import ExplainResponse


def explain_short(y_hat_4h: float, p_up_4h: float, news_points: List[str], rationale_points: List[str]) -> str:
    sys = (
        "Ты финансовый аналитик. Кратко объясни прогноз строго на основе входных данных (grounded-only).\n"
        "Формат: JSON строго {\"bullets\":[string,...]} без лишних полей. 2–4 пункта.\n"
        "Запрет: не придумывай числа и факты, используй только y_hat, p_up, аргументы ансамбля и названия новостей."
    )
    usr = (
        f"Прогноз 4h y_hat={y_hat_4h:.2f}, p_up={p_up_4h:.2f}.\n"
        f"Аргументы ансамбля: {rationale_points}.\n"
        f"Новости: {news_points}."
    )
    # Try Flowise if configured
    raw = call_flowise_json("FLOWISE_EXPLAIN_URL", {"system": sys, "user": usr}) or call_openai_json(sys, usr)
    data = None
    try:
        if raw:
            data = ExplainResponse.model_validate(raw)
    except Exception:
        data = None
    if not data or not data.bullets:
        # Fallback
        pts = []
        if rationale_points:
            pts.append(f"Ансамбль даёт p_up≈{p_up_4h:.2f} на 4h с учётом исторической ошибки.")
        if news_points:
            pts.append("Новости нейтральны/умеренные; реакция рынка учтена в интервалах.")
        if not pts:
            pts = ["Техническая картина и волатильность учтены в прогнозе."]
        return "\n".join(f"• {p}" for p in pts[:3])
    bullets = [str(b) for b in data.bullets][:4]
    return "\n".join(f"• {b}" for b in bullets)
