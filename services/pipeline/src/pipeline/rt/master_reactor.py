from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger

from ..infra.logging_config import init_logging
from ..infra.obs import init_sentry
from ..infra.db import (
    upsert_features_snapshot,
    upsert_regime,
    upsert_prediction,
    upsert_ensemble_weights,
    upsert_explanations,
    upsert_trade_suggestion,
)
from ..mcp.client import call_tool
from ..reasoning.chart_reasoning import ChartReasoningInput as _CRInput
from ..reasoning.chart_reasoning import run as run_chart_reasoning
from ..trading.trade_recommend import TradeRecommendInput
from ..trading.trade_recommend import run as run_trade
from ..trading.verifier import verify
from ..trading.publish_telegram import publish_message
from ..data.ingest_prices import IngestPricesInput, run as run_prices
from ..data.ingest_news import IngestNewsInput, run as run_news
from ..features.features_calc import FeaturesCalcInput, run as run_features
from .queue import pop_trigger


@dataclass
class Trigger:
    type: str
    symbol: str
    ts: int
    meta: Dict[str, Any]


def _parse_trigger(ev: Dict[str, Any]) -> Optional[Trigger]:
    try:
        return Trigger(
            type=str(ev.get("type") or "").upper(),
            symbol=str(ev.get("symbol") or os.getenv("PAPER_SYMBOL", "BTC/USDT")),
            ts=int(ev.get("ts") or 0),
            meta=dict(ev.get("meta") or {}),
        )
    except Exception:
        return None


def _horizon_for(t: Trigger) -> int:
    # Derive horizon minutes from trigger type
    mapping = {
        "VOL_SPIKE": int(os.getenv("RT_HORIZON_VOL_SPIKE_MIN", "30")),
        "DELTA_SPIKE": int(os.getenv("RT_HORIZON_DELTA_SPIKE_MIN", "30")),
        "NEWS": int(os.getenv("RT_HORIZON_NEWS_MIN", "60")),
    }
    return int(mapping.get(t.type, int(os.getenv("RT_HORIZON_DEFAULT_MIN", "30"))))


def _short_reason(t: Trigger) -> str:
    if t.type == "VOL_SPIKE":
        return f"Триггер: всплеск объёма (x{float(t.meta.get('ratio', 0.0)):.1f})"
    if t.type == "DELTA_SPIKE":
        return f"Триггер: дисбаланс покупок/продаж (Δ≈{float(t.meta.get('delta_agg', 0.0)):.2f})"
    if t.type == "NEWS":
        return f"Триггер: новость ({t.meta.get('source','')}, score={float(t.meta.get('impact_score',0.0)):.2f})"
    return f"Триггер: {t.type}"


def run_realtime_analysis(trigger: Trigger) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ_rt")
    slot = f"rt:{trigger.type.lower()}"

    # 1) Ingest recent prices + news (shorter windows)
    end_ts = int(now.timestamp())
    prices = run_prices(
        IngestPricesInput(
            run_id=run_id,
            slot=slot,
            symbols=[trigger.symbol.replace("/", "")],
            start_ts=end_ts - int(os.getenv("RT_PRICES_LOOKBACK_SEC", "172800")),  # 48h
            end_ts=end_ts,
            provider=os.getenv("CCXT_PROVIDER", "binance"),
            timeframe=os.getenv("RT_TIMEFRAME", "1m"),
        )
    )
    news = None
    try:
        news = run_news(
            IngestNewsInput(
                run_id=run_id,
                slot=slot,
                time_window_hours=int(os.getenv("RT_NEWS_WINDOW_H", "6")),
                query=os.getenv("NEWS_QUERY", "bitcoin OR BTC"),
            )
        )
    except Exception:
        news = None
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

    # 3) Tools via MCP
    regime = call_tool("run_regime_detection", {"features_s3": features_s3}) or {}
    upsert_regime(
        run_id,
        str(regime.get("label", "range")),
        float(regime.get("confidence", 0.0) or 0.0),
        regime.get("features", {}),
    )
    neighbors = call_tool("run_similarity_search", {"features_s3": features_s3}) or {}
    topk = neighbors.get("topk") or []
    hmin = _horizon_for(trigger)
    hz = f"{hmin}m"
    out = call_tool("run_models_and_ensemble", {"features_s3": features_s3, "horizon": hz}) or {}
    if not out:
        try:
            from ..models.models import ModelsInput as _MIn, run as _run_models
            from ..ensemble.ensemble_rank import EnsembleInput as _EIn, run as _run_ens

            mm = _run_models(_MIn(features_path_s3=features_s3, horizon_minutes=hmin))
            preds = [p.model_dump() for p in mm.preds]
            ee = _run_ens(_EIn(preds=preds, horizon=hz))
            out = {
                "ensemble": {
                    "y_hat": float(ee.y_hat),
                    "interval": [float(ee.interval[0]), float(ee.interval[1])],
                    "proba_up": float(ee.proba_up),
                    "weights": ee.weights,
                    "rationale_points": ee.rationale_points,
                },
                "last_price": float(mm.last_price),
                "atr": float(mm.atr),
                "preds": preds,
            }
        except Exception:
            out = {}

    e = (out.get("ensemble") or {})
    preds = out.get("preds") or []
    last_price = float(out.get("last_price", 0.0) or 0.0)
    atr = float(out.get("atr", 0.0) or 0.0)
    upsert_prediction(
        run_id,
        hz,
        float(e.get("y_hat", 0.0) or 0.0),
        float((e.get("interval") or [0.0, 0.0])[0]),
        float((e.get("interval") or [0.0, 0.0])[1]),
        float(e.get("proba_up", 0.5) or 0.5),
        {p.get("model"): {k: p.get(k) for k in ("y_hat", "pi_low", "pi_high", "proba_up", "cv_metrics")} for p in preds},
    )
    upsert_ensemble_weights(run_id, e.get("weights") or {})

    # 4) Chart reasoning (LLM-based technical view)
    try:
        ta = run_chart_reasoning(_CRInput(features_path_s3=features_s3))
    except Exception:
        ta = {"technical_sentiment": "neutral", "confidence_score": 0.5, "key_observations": []}

    # 5) Trade card
    tri = TradeRecommendInput(
        current_price=last_price,
        y_hat_4h=float(e.get("y_hat", 0.0) or 0.0),  # reusing field name for short horizon
        interval_low=float((e.get("interval") or [0.0, 0.0])[0]),
        interval_high=float((e.get("interval") or [0.0, 0.0])[1]),
        proba_up=float(e.get("proba_up", 0.5) or 0.5),
        atr=atr,
        regime=str(regime.get("label")),
        account_equity=float(os.getenv("ACCOUNT_EQUITY", "1000")),
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.005")),
        leverage_cap=float(os.getenv("RISK_LEVERAGE_CAP", "25")),
        now_ts=int(now.timestamp()),
        tz_name=os.getenv("TIMEZONE", "Europe/Moscow"),
        valid_for_minutes=int(os.getenv("RT_VALID_FOR_MIN", "30")),
        horizon_minutes=hmin,
    )
    card = run_trade(tri)
    # annotate reason codes
    rc = card.get("reason_codes") or []
    rc.append(f"TRIGGER:{trigger.type}")
    card["reason_codes"] = rc
    ok, reason = verify(
        card,
        current_price=tri.current_price,
        leverage_cap=tri.leverage_cap,
        interval=(tri.interval_low, tri.interval_high),
        atr=tri.atr,
    )
    upsert_trade_suggestion(run_id, card)

    # 6) Explain (short note)
    reason_text = _short_reason(trigger)
    md = (
        f"<b>Real-Time сигнал</b> ({hz})\n"
        f"{reason_text}\n\n"
        + (f"Тех.сентимент: {ta.get('technical_sentiment','n/a')} (conf≈{float(ta.get('confidence_score',0.0)):.2f})\n" if ta else "")
        + f"p_up≈{float(e.get('proba_up',0.5)):.2f}, ŷ≈{float(e.get('y_hat',0.0)):.2f}\n"
        + f"Рекомендация: {card.get('side')} lev×{card.get('leverage')} SL={card.get('stop_loss')} TP={card.get('take_profit')}\n"
        + f"Горизонт: ~{hmin} мин"
    )
    upsert_explanations(run_id, md, [])

    # Optional: publish concise Telegram alert
    try:
        publish_message(md)
    except Exception:
        pass

    return {
        "run_id": run_id,
        "slot": slot,
        "horizon": hz,
        "trigger": trigger.type,
        "trade": {"card": card, "verified": ok, "reason": reason},
    }


def main() -> None:
    init_logging()
    try:
        init_sentry()
    except Exception:
        pass
    logger.info("RT MasterReactor started; waiting for triggers…")
    while True:
        ev = pop_trigger(timeout_s=5)
        if not ev:
            continue
        t = _parse_trigger(ev)
        if not t:
            continue
        try:
            out = run_realtime_analysis(t)
            logger.info({k: (str(v)[:120] + "…" if isinstance(v, str) and len(v) > 120 else v) for k, v in out.items()})
        except Exception as e:  # noqa: BLE001
            logger.exception(f"RT analysis failed for trigger {t.type}: {e}")


if __name__ == "__main__":
    main()
