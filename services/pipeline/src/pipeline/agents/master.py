from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List

from loguru import logger

from ..infra.logging_config import init_logging
from ..infra.obs import init_sentry
from ..infra.metrics import push_values
from ..infra.db import (
    upsert_features_snapshot,
    upsert_regime,
    upsert_prediction,
    upsert_ensemble_weights,
    upsert_explanations,
    upsert_trade_suggestion,
    insert_run_summary,
    fetch_latest_strategic_verdict,
    get_data_source_trust,
)
from ..infra.s3 import download_bytes
from ..mcp.client import call_tool
from ..reporting.charts import plot_price_with_levels
from ..reasoning.debate_arbiter import multi_debate
from ..reasoning.explain import explain_short
from ..trading.trade_recommend import TradeRecommendInput
from ..trading.trade_recommend import run as run_trade
from ..trading.verifier import verify

# Reuse ingestion and feature builders directly for now
from ..data.ingest_prices import IngestPricesInput, run as run_prices
from ..data.ingest_news import IngestNewsInput, run as run_news
from ..features.features_calc import FeaturesCalcInput, run as run_features
from ..reasoning.chart_reasoning import ChartReasoningInput as _CRInput
from ..reasoning.chart_reasoning import run as run_chart_reasoning
from .chart_vision_agent import ChartVisionAgent
from .technical_synthesis_agent import TechnicalSynthesisAgent
from .memory_guardian_agent import MemoryGuardianAgent


@dataclass
class MasterContext:
    run_id: str
    slot: str


def run_master_flow(slot: str = "manual") -> Dict[str, Any]:
    """MasterAgent: orchestrates the forecasting flow via MCP tools + local helpers.

    Steps:
    - Ingest prices + news, compute features
    - Use MCP tools: regime, similar, models+ensemble (4h/12h)
    - Multi-agent debate, explain
    - Trade card recommendation and verification
    - Persist summary for memory
    """
    init_logging()
    init_sentry()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    ctx = MasterContext(run_id=run_id, slot=slot)
    vision_agent = ChartVisionAgent()
    synthesis_agent = TechnicalSynthesisAgent()
    memory_guardian = MemoryGuardianAgent()

    # 1) Ingest prices + news
    end_ts = int(now.timestamp())
    start_ts = end_ts - 72 * 3600
    prices = run_prices(
        IngestPricesInput(
            run_id=run_id,
            slot=slot,
            symbols=["BTCUSDT"],
            start_ts=start_ts,
            end_ts=end_ts,
            provider=os.getenv("CCXT_PROVIDER", "binance"),
            timeframe=os.getenv("CCXT_TIMEFRAME", "1m"),
        )
    )
    news = run_news(
        IngestNewsInput(
            run_id=run_id,
            slot=slot,
            time_window_hours=int(os.getenv("NEWS_WINDOW_H", "12")),
            query=os.getenv("NEWS_QUERY", "bitcoin OR BTC"),
        )
    )
    news_signals = [s.model_dump() for s in getattr(news, "news_signals", [])] if news else []
    news_facts = [f for f in (getattr(news, "news_facts", []) or [])]
    source_trust: Dict[str, float] = {}
    for sig in news_signals:
        try:
            src_name = str(sig.get("source") or sig.get("provider") or "").strip()
        except Exception:
            src_name = ""
        if not src_name or src_name in source_trust:
            continue
        trust_val = get_data_source_trust(src_name)
        if trust_val is None and sig.get("provider") and src_name.lower() != str(sig["provider"]).lower():
            trust_val = get_data_source_trust(str(sig.get("provider")))
        if trust_val is not None:
            source_trust[src_name] = float(trust_val)

    # 2) Features
    feats = run_features(
        FeaturesCalcInput(
            prices_path_s3=getattr(prices, "prices_path_s3", ""),
            news_signals=news_signals,
            news_facts=news_facts,
            run_id=run_id,
            slot=slot,
        )
    )
    features_s3 = getattr(feats, "features_path_s3", "")
    upsert_features_snapshot(
        run_id, getattr(feats, "snapshot_ts", now.isoformat()), features_s3
    )
    indicators_snapshot = _latest_indicator_snapshot(features_s3)
    smc_snapshot = fetch_latest_strategic_verdict("SMC Analyst", "BTC/USDT") or {}

    # 3) MCP tools
    regime = call_tool("run_regime_detection", {"features_s3": features_s3}) or {}
    upsert_regime(run_id, str(regime.get("label", "range")), float(regime.get("confidence", 0.0) or 0.0), regime.get("features", {}))
    neighbors = call_tool("run_similarity_search", {"features_s3": features_s3}) or {}
    topk = neighbors.get("topk") or []
    out4 = call_tool("run_models_and_ensemble", {"features_s3": features_s3, "horizon": "4h"}) or {}
    out12 = call_tool("run_models_and_ensemble", {"features_s3": features_s3, "horizon": "12h"}) or {}

    # Persist predictions
    e4 = (out4.get("ensemble") or {})
    preds4 = out4.get("preds") or []
    upsert_prediction(
        run_id,
        "4h",
        float(e4.get("y_hat", 0.0) or 0.0),
        float((e4.get("interval") or [0.0, 0.0])[0]),
        float((e4.get("interval") or [0.0, 0.0])[1]),
        float(e4.get("proba_up", 0.5) or 0.5),
        {p.get("model"): {k: p.get(k) for k in ("y_hat", "pi_low", "pi_high", "proba_up", "cv_metrics")} for p in preds4},
    )
    try:
        upsert_ensemble_weights(run_id, e4.get("weights") or {})
    except Exception:
        pass

    e12 = (out12.get("ensemble") or {})
    preds12 = out12.get("preds") or []
    upsert_prediction(
        run_id,
        "12h",
        float(e12.get("y_hat", 0.0) or 0.0),
        float((e12.get("interval") or [0.0, 0.0])[0]),
        float((e12.get("interval") or [0.0, 0.0])[1]),
        float(e12.get("proba_up", 0.5) or 0.5),
        {p.get("model"): {k: p.get(k) for k in ("y_hat", "pi_low", "pi_high", "proba_up", "cv_metrics")} for p in preds12},
    )

    # 4) Chart reasoning (LLM-based technical view)
    try:
        ta = run_chart_reasoning(_CRInput(features_path_s3=features_s3))
    except Exception:
        ta = {"technical_sentiment": "neutral", "confidence_score": 0.5, "key_observations": []}
    if hasattr(ta, "model_dump"):
        ta = ta.model_dump()
    elif not isinstance(ta, dict):
        ta = dict(ta or {})

    # 5) Debate + explain
    try:
        news_top = [f"{s.get('title')} ({s.get('source')})" for s in news_signals[:3]]
    except Exception:
        news_top = []
    memory_items = call_tool("get_recent_run_summaries", {"n": 3}) or {"items": []}
    mem = [
        f"{it.get('created_at')} run={it.get('run_id')} p_up4h={((it.get('final') or {}).get('e4') or {}).get('proba_up', '')}"
        for it in (memory_items.get("items") or [])
    ]
    # Append compressed lessons (if any) for richer long-term memory context
    try:
        lessons_resp = call_tool("get_lessons", {"n": 3}) or {"items": []}
        for ls in lessons_resp.get("items") or []:
            mem.append(f"lesson: {ls.get('lesson_text')}")
    except Exception:
        pass
    planned_signal = "BULLISH" if float(e4.get("y_hat", 0.0) or 0.0) >= 0 else "BEARISH"
    guardian_lessons: List[dict] = []
    try:
        mg_result = memory_guardian.query(
            {
                "scope": "trading",
                "context": {
                    "planned_signal": planned_signal,
                    "market_regime": str(regime.get("label", "range")),
                    "probability_up": float(e4.get("proba_up", 0.5) or 0.5),
                },
                "top_k": 3,
            }
        )
        if mg_result and isinstance(mg_result.output, dict):
            guardian_lessons = mg_result.output.get("lessons", []) or []
    except Exception:
        guardian_lessons = []
    chart_s3 = None
    vision_output = {"bias": "neutral", "confidence": 0.5, "insights": []}
    synthesis_output = {"verdict": {"bias": "NEUTRAL", "confidence": 0.0, "score": 0.0}, "components": []}
    try:
        chart_s3 = plot_price_with_levels(
            features_path_s3=features_s3,
            title=f"BTC {slot} — 4h/12h прогноз",
            y_hat_4h=float(e4.get("y_hat", 0.0) or 0.0),
            y_hat_12h=float(e12.get("y_hat", 0.0) or 0.0),
            levels=[],
            slot=slot,
        )
    except Exception:
        pass

    if chart_s3:
        try:
            vision_result = vision_agent.run(
                {
                    "run_id": run_id,
                    "symbol": "BTCUSDT",
                    "image_urls": [chart_s3],
                    "regime": str(regime.get("label", "range")),
                }
            )
            vision_output = vision_result.output
        except Exception:
            vision_output = {"bias": "neutral", "confidence": 0.4, "insights": []}

    try:
        synthesis_output = synthesis_agent.run(
            {
                "run_id": run_id,
                "symbol": "BTCUSDT",
                "indicators": indicators_snapshot,
                "smc": smc_snapshot,
                "vision": vision_output,
                "features_meta": {"features_s3": features_s3},
            }
        ).output
    except Exception:
        synthesis_output = {"verdict": {"bias": "NEUTRAL", "confidence": 0.0, "score": 0.0}, "components": []}

    if vision_output:
        mem.append(
            f"chart_vision bias={vision_output.get('bias')} conf={vision_output.get('confidence')}"
        )
    if synthesis_output:
        verdict = synthesis_output.get("verdict", {})
        mem.append(
            f"tech_synthesis bias={verdict.get('bias')} conf={verdict.get('confidence')} score={verdict.get('score')}"
        )
    tech_sentiment = ta.get("technical_sentiment")
    if tech_sentiment:
        mem.append(f"chart_reasoning sentiment={tech_sentiment}")

    ta_context = dict(ta)
    ta_context["technical_synthesis"] = synthesis_output
    ta_context["vision"] = vision_output
    deb_text, risk_flags = multi_debate(
        regime=str(regime.get("label")),
        news_top=news_top,
        neighbors=topk,
        memory=mem,
        trust=(e4.get("weights") or {}),
        ta=ta_context,
        lessons=guardian_lessons,
        source_trust=source_trust,
    )
    expl_text = explain_short(
        float(e4.get("y_hat", 0.0) or 0.0), float(e4.get("proba_up", 0.5) or 0.5), news_top, e4.get("rationale_points") or []
    )

    # 6) Trade card
    last_price = float(out4.get("last_price", 0.0) or 0.0)
    atr = float(out4.get("atr", 0.0) or 0.0)
    safe_mode = os.getenv("SAFE_MODE", "0") in {"1", "true", "True"}
    try:
        overrides = json.loads(os.getenv("SAFE_MODE_RISK_OVERRIDES", "{}"))
    except Exception:
        overrides = {}
    base_risk = float(os.getenv("RISK_PER_TRADE", "0.005"))
    base_leverage = float(os.getenv("RISK_LEVERAGE_CAP", "25"))
    if safe_mode:
        if "risk_per_trade" in overrides:
            base_risk = min(base_risk, float(overrides["risk_per_trade"]))
        if "leverage_cap" in overrides:
            base_leverage = min(base_leverage, float(overrides["leverage_cap"]))
    tri = TradeRecommendInput(
        current_price=last_price,
        y_hat_4h=float(e4.get("y_hat", 0.0) or 0.0),
        interval_low=float((e4.get("interval") or [0.0, 0.0])[0]),
        interval_high=float((e4.get("interval") or [0.0, 0.0])[1]),
        proba_up=float(e4.get("proba_up", 0.5) or 0.5),
        atr=atr,
        regime=str(regime.get("label")),
        account_equity=float(os.getenv("ACCOUNT_EQUITY", "1000")),
        risk_per_trade=base_risk,
        leverage_cap=base_leverage,
        now_ts=int(now.timestamp()),
        tz_name=os.getenv("TIMEZONE", "Europe/Moscow"),
        valid_for_minutes=90,
        horizon_minutes=240,
    )
    card = run_trade(tri)
    ok, reason = verify(
        card,
        current_price=tri.current_price,
        leverage_cap=tri.leverage_cap,
        interval=(tri.interval_low, tri.interval_high),
        atr=tri.atr,
    )
    upsert_trade_suggestion(run_id, card)

    # 7) Persist explanation, memory
    md = (
        "<b>Почему так</b>\n"
        + expl_text
        + "\n\n<b>Арбитраж</b>\n"
        + deb_text
        + "\n\n<b>Technical synthesis</b>\n"
        + f"Bias: {synthesis_output.get('verdict', {}).get('bias', 'NEUTRAL')} "
        + f"({synthesis_output.get('verdict', {}).get('confidence', 0.0):.2f})"
        + (f"\nChart: {chart_s3}" if chart_s3 else "")
    )
    upsert_explanations(run_id, md, risk_flags)

    final_summary = {
        "slot": slot,
        "regime": str(regime.get("label")),
        "regime_conf": float(regime.get("confidence", 0.0) or 0.0),
        "ta": ta,
        "e4": {
            "y_hat": float(e4.get("y_hat", 0.0) or 0.0),
            "proba_up": float(e4.get("proba_up", 0.5) or 0.5),
            "interval": [
                float((e4.get("interval") or [0.0, 0.0])[0]),
                float((e4.get("interval") or [0.0, 0.0])[1]),
            ],
        },
        "e12": {
            "y_hat": float(e12.get("y_hat", 0.0) or 0.0),
            "proba_up": float(e12.get("proba_up", 0.5) or 0.5),
            "interval": [
                float((e12.get("interval") or [0.0, 0.0])[0]),
                float((e12.get("interval") or [0.0, 0.0])[1]),
            ],
        },
        "risk_flags": risk_flags,
        "vision": vision_output,
        "synthesis": synthesis_output,
        "smc": smc_snapshot,
        "indicators": indicators_snapshot,
        "safe_mode": safe_mode,
        "risk_overrides": overrides,
    }
    insert_run_summary(run_id, final_summary, None)

    # 7) Metrics snapshot
    try:
        push_values(
            job="master_agent",
            values={
                "p_up_4h": final_summary["e4"]["proba_up"],
                "p_up_12h": final_summary["e12"]["proba_up"],
            },
            labels={"slot": slot},
        )
    except Exception:
        pass

    return {
        "run_id": run_id,
        "slot": slot,
        "regime": regime,
        "neighbors": topk,
        "ensemble_4h": e4,
        "ensemble_12h": e12,
        "trade": {"card": card, "verified": ok, "reason": reason},
        "explain": md,
    }


def _latest_indicator_snapshot(features_s3: str) -> Dict[str, float]:
    if not features_s3:
        return {}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        raw = download_bytes(features_s3)
        cols = [
            "ts",
            "close",
            "ema_20",
            "ema_50",
            "rsi_dynamic",
            "atr_dynamic",
            "bb_width_dynamic",
            "indicator_cfg_rsi",
            "indicator_cfg_atr",
            "indicator_cfg_bb_window",
            "indicator_cfg_bb_std",
        ]
        table = pq.read_table(pa.BufferReader(raw), columns=cols)
        df = table.to_pandas().sort_values("ts").reset_index(drop=True)
        if df.empty:
            return {}
        row = df.iloc[-1]
        close = float(row.get("close", 0.0) or 0.0)
        ema_fast = float(row.get("ema_20", 0.0) or 0.0)
        ema_slow = float(row.get("ema_50", 0.0) or 0.0)
        trend_strength = (ema_fast - ema_slow) / (abs(close) + 1e-9)
        return {
            "rsi_dynamic": float(row.get("rsi_dynamic", 50.0) or 50.0),
            "atr_dynamic": float(row.get("atr_dynamic", 0.0) or 0.0),
            "bb_width_dynamic": float(row.get("bb_width_dynamic", 0.0) or 0.0),
            "trend_strength": float(trend_strength),
            "cfg_rsi": int(row.get("indicator_cfg_rsi", 14) or 14),
            "cfg_atr": int(row.get("indicator_cfg_atr", 14) or 14),
            "cfg_bb_window": int(row.get("indicator_cfg_bb_window", 20) or 20),
            "cfg_bb_std": float(row.get("indicator_cfg_bb_std", 2.0) or 2.0),
        }
    except Exception:
        return {}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", default=os.environ.get("SLOT", "manual"))
    args = parser.parse_args()
    out = run_master_flow(slot=args.slot)
    logger.info({k: (str(v)[:120] + "…" if isinstance(v, str) and len(v) > 120 else v) for k, v in out.items()})


if __name__ == "__main__":
    main()
