from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

from loguru import logger

from ..reasoning.llm import call_openai_json, call_flowise_json
from ..mcp.client import call_tool as mcp_call
from ..reasoning.schemas import ScenarioResponse, ScenarioItem
from .scenario_modeler import run as fallback_scenarios, ScenarioModelerInput


def run_llm_or_fallback(
    features_path_s3: str,
    current_price: float,
    atr: float,
    slot: str = "manual",
    onchain_context: Optional[Dict[str, Any]] = None,
    macro_flags: Optional[List[str]] = None,
    event_hypotheses: Optional[List[dict]] = None,
) -> Tuple[List[dict], List[float]]:
    # Ground facts via MCP tools (optional)
    facts = {}
    try:
        # last features row basics
        tail = mcp_call("get_features_tail", {"features_s3": features_path_s3, "n": 1}) or {}
        levels = mcp_call("levels_quantiles", {"features_s3": features_path_s3, "qs": [0.2, 0.5, 0.8]}) or {}
        facts = {"tail": tail, "levels": levels}
    except Exception:
        facts = {}

    sys = (
        "Ты помощник по сценариям BTC. Сформируй 5 веток (grounded-only), строго JSON: "
        "{\"scenarios\":[{\"if_level\":string,\"then_path\":string,\"prob\":number,\"invalidation\":string}],\"levels\":[number,...]}.\n"
        "Правила: вероятности суммарно ≈1; не противоречь риск‑политике (SL/TP/плечо ≤25x); не придумывай числа — используй близкие уровни к текущей цене и ATR‑масштаб."
    )
    oc = onchain_context or {}
    mf = macro_flags or []
    usr = (
        f"slot={slot}; price={current_price:.2f}; atr={atr:.2f}.\n"
        f"facts={facts}.\n"
        f"onchain={oc}. macro_flags={mf}. events={event_hypotheses or []}. Сформулируй 5 коротких веток с вероятностями и инвалидацией, опираясь только на facts/ончейн/макро/события."
    )
    raw = call_flowise_json("FLOWISE_SCENARIO_URL", {"system": sys, "user": usr}) or call_openai_json(sys, usr)
    data = None
    try:
        if raw:
            data = ScenarioResponse.model_validate(raw)
    except Exception:
        data = None
    if not data or not data.scenarios:
        logger.warning("scenario_helper: LLM unavailable, using heuristic fallback")
        return fallback_scenarios(ScenarioModelerInput(features_path_s3=features_path_s3, current_price=current_price, atr=atr, slot=slot))
    scenarios = [s.model_dump() for s in data.scenarios]
    levels = data.levels or []
    # sanitize
    out = []
    for s in scenarios[:5]:
        try:
            item = ScenarioItem.model_validate(s)
            out.append(item.model_dump())
        except Exception:
            continue
    if not out:
        return fallback_scenarios(ScenarioModelerInput(features_path_s3=features_path_s3, current_price=current_price, atr=atr, slot=slot))
    # normalize probs
    total = sum(s["prob"] for s in out) or 1.0
    for s in out:
        s["prob"] = round(s["prob"] / total, 2)
    return out, levels
