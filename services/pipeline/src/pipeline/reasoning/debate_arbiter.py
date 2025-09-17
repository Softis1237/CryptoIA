from __future__ import annotations

from typing import Dict, List, Optional
import os

from .llm import call_flowise_json, call_openai_json
from ..infra.db import fetch_agent_config
from .schemas import DebateResponse


def debate(
    rationale_points: List[str],
    regime: Optional[str],
    news_top: List[str],
    neighbors: Optional[List[dict]] = None,
    memory: Optional[List[str]] = None,
    trust: Optional[Dict[str, float]] = None,
    ta: Optional[dict] = None,
    lessons: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, List[str]]:
    lessons_txt = ""
    try:
        if lessons:
            items = [f"- {l.get('title')}: {l.get('action') or l.get('insight','')[:80]}" for l in lessons[:5]]
            if items:
                lessons_txt = "\n".join(items)
    except Exception:
        lessons_txt = ""

    sys = (
        "You are an arbiter: distill the arguments into 3–6 grounded bullet points.\n"
        'Return JSON exactly as {"bullets":[string,...],"risk_flags":[string,...]} with no extra fields.\n'
        "Base only on model arguments, regime, top news and similar windows; do not invent numbers.\n"
        "If memory of previous releases is provided — consider it as context; if trust weights are provided — use them to prioritize arguments."
        + ("\nConsider the following lessons learned from similar situations and avoid repeating mistakes:\n" + lessons_txt if lessons_txt else "")
    )
    usr = (
        f"Model arguments: {rationale_points}\n"
        f"Regime: {regime}\n"
        f"Top news: {news_top}\n"
        f"Similar windows: {neighbors}\n"
        f"Memory: {memory or []}\n"
        f"Trust: {trust or {}}\n"
        f"TA: {ta or {}}"
    )
    raw = call_flowise_json("FLOWISE_DEBATE_URL", {"system": sys, "user": usr})
    if not raw or raw.get("status") == "error":
        # Prefer dedicated debate model if provided (DB-configurable)
        cfg = fetch_agent_config("DebateArbiter") or {}
        params = cfg.get("parameters") or {}
        model = params.get("model") if isinstance(params, dict) else os.getenv("OPENAI_MODEL_DEBATE")
        raw = call_openai_json(cfg.get("system_prompt") or sys, usr, model=model)
    data = None
    try:
        if raw and raw.get("status") != "error":
            data = DebateResponse.model_validate(raw)
    except Exception:
        data = None
    if not data:
        # Fallback: concise rephrase of rationale_points
        bullets = [p for p in rationale_points][:4] or ["Model signals are neutral."]
        return "\n".join(f"• {b}" for b in bullets), []
    bullets = [str(b) for b in data.bullets][:6]
    flags = [str(r) for r in data.risk_flags][:6]
    return "\n".join(f"• {b}" for b in bullets), flags


def multi_debate(
    regime: Optional[str],
    news_top: List[str],
    neighbors: Optional[List[dict]] = None,
    memory: Optional[List[str]] = None,
    trust: Optional[Dict[str, float]] = None,
    model_bull: Optional[str] = None,
    model_bear: Optional[str] = None,
    model_quant: Optional[str] = None,
    ta: Optional[dict] = None,
) -> tuple[str, List[str]]:
    """Run a lightweight multi-persona debate and aggregate with the arbiter.

    Returns (markdown_bullets, risk_flags).
    """
    bull_sys = (
        "Ты бычий аналитик. Найди убедительные аргументы за рост цены. "
        "Думай последовательно, но в ответе верни только JSON {\"bullets\":[...]}"
    )
    bear_sys = (
        "Ты медвежий аналитик. Найди сильные аргументы за падение цены. "
        "Думай последовательно, но в ответе верни только JSON {\"bullets\":[...]}"
    )
    quant_sys = (
        "Ты квант. Игнорируй эмоции, сосредоточься на цифрах и волатильности. "
        "Верни только JSON {\"bullets\":[...]}"
    )
    ctx = (
        f"Regime: {regime}\nTop news: {news_top}\nSimilar windows: {neighbors}\nMemory: {memory}\nTrust: {trust}\nTA: {ta}"
    )
    out_bull = call_openai_json(bull_sys, ctx, model=model_bull or os.getenv("OPENAI_MODEL_MASTER") or os.getenv("OPENAI_MODEL_DEBATE"))
    out_bear = call_openai_json(bear_sys, ctx, model=model_bear or os.getenv("OPENAI_MODEL_MASTER") or os.getenv("OPENAI_MODEL_DEBATE"))
    out_quant = call_openai_json(quant_sys, ctx, model=model_quant or os.getenv("OPENAI_MODEL_MASTER") or os.getenv("OPENAI_MODEL_DEBATE"))
    bullets_all = []
    for raw in [out_bull, out_bear, out_quant]:
        try:
            bullets_all.extend([str(b) for b in (raw.get("bullets") or raw.get("points") or [])])
        except Exception:
            continue
    if not bullets_all:
        bullets_all = [
            "Модели не предоставили разногласий — используем базовые доводы по сигналам.",
        ]
    # Feed into arbiter
    text, flags = debate(bullets_all, regime, news_top, neighbors, memory, trust, ta)
    return text, flags
