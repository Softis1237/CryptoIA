from __future__ import annotations

from typing import List

from .llm import call_flowise_json, call_openai_json
from .schemas import ExplainResponse


def explain_short(
    y_hat_4h: float, p_up_4h: float, news_points: List[str], rationale_points: List[str]
) -> str:
    sys = (
        "You are a financial analyst. Briefly explain the forecast strictly grounded in the inputs (no fabrication).\n"
        'Return JSON exactly as {"bullets":[string,...]} with 2–4 concise bullet points and no extra fields.\n'
        "Do not invent numbers or facts; use only y_hat, p_up, ensemble rationale, and news titles."
    )
    usr = (
        f"Forecast 4h y_hat={y_hat_4h:.2f}, p_up={p_up_4h:.2f}.\n"
        f"Ensemble rationale: {rationale_points}.\n"
        f"Top news: {news_points}."
    )
    # Try Flowise if configured
    raw = call_flowise_json("FLOWISE_EXPLAIN_URL", {"system": sys, "user": usr})
    if not raw or raw.get("status") == "error":
        raw = call_openai_json(sys, usr)
    data = None
    try:
        if raw and raw.get("status") != "error":
            data = ExplainResponse.model_validate(raw)
    except Exception:
        data = None
    if not data or not data.bullets:
        # Fallback
        pts = []
        if rationale_points:
            pts.append(
                f"Ensemble suggests p_up≈{p_up_4h:.2f} for 4h, calibrated by historical error."
            )
        if news_points:
            pts.append(
                "News are neutral/moderate; market reaction is reflected in the intervals."
            )
        if not pts:
            pts = ["Technical structure and volatility are reflected in the forecast."]
        return "\n".join(f"• {p}" for p in pts[:3])
    bullets = [str(b) for b in data.bullets][:4]
    return "\n".join(f"• {b}" for b in bullets)
