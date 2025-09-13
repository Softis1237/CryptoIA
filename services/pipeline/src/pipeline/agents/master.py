from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

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
)
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
    deb_text, risk_flags = multi_debate(
        regime=str(regime.get("label")),
        news_top=news_top,
        neighbors=topk,
        memory=mem,
        trust=(e4.get("weights") or {}),
        ta=ta,
    )
    expl_text = explain_short(
        float(e4.get("y_hat", 0.0) or 0.0), float(e4.get("proba_up", 0.5) or 0.5), news_top, e4.get("rationale_points") or []
    )

    chart_s3 = None
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

    # 6) Trade card
    last_price = float(out4.get("last_price", 0.0) or 0.0)
    atr = float(out4.get("atr", 0.0) or 0.0)
    tri = TradeRecommendInput(
        current_price=last_price,
        y_hat_4h=float(e4.get("y_hat", 0.0) or 0.0),
        interval_low=float((e4.get("interval") or [0.0, 0.0])[0]),
        interval_high=float((e4.get("interval") or [0.0, 0.0])[1]),
        proba_up=float(e4.get("proba_up", 0.5) or 0.5),
        atr=atr,
        regime=str(regime.get("label")),
        account_equity=float(os.getenv("ACCOUNT_EQUITY", "1000")),
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.005")),
        leverage_cap=float(os.getenv("RISK_LEVERAGE_CAP", "25")),
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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", default=os.environ.get("SLOT", "manual"))
    args = parser.parse_args()
    out = run_master_flow(slot=args.slot)
    logger.info({k: (str(v)[:120] + "…" if isinstance(v, str) and len(v) > 120 else v) for k, v in out.items()})


if __name__ == "__main__":
    main()
