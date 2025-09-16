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
from ..infra.db import fetch_model_trust_regime_event, fetch_model_trust_regime
from ..infra.db import fetch_latest_strategic_verdict
from ..mcp.client import call_tool
from ..reasoning.chart_reasoning import ChartReasoningInput as _CRInput
from ..reasoning.chart_reasoning import run as run_chart_reasoning
from ..trading.trade_recommend import TradeRecommendInput
from ..trading.trade_recommend import run as run_trade
from ..trading.verifier import verify
from ..trading.confidence import compute as compute_conf
from ..telegram_bot.publisher import publish_message, publish_message_to
from ..data.ingest_prices import IngestPricesInput, run as run_prices
from ..data.ingest_news import IngestNewsInput, run as run_news
from ..features.features_calc import FeaturesCalcInput, run as run_features
from .queue import pop_trigger
from ..infra.metrics import push_values
from ..utils.calibration import calibrate_proba
from .queue import get_key as _get_cache, setex as _set_cache
from ..infra.health import start_background as start_health_server
from .alert_priority import mark_and_score as _alert_prio
from ..agents.alpha_hunter import (
    match_strategies_on_features as _alpha_match,
    AlphaMatchResult,
)


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


def _classify_news_event(title: str) -> str:
    s = (title or "").upper()
    if "ETF" in s:
        return "ETF_APPROVAL"
    if "SEC" in s or "LAWSUIT" in s or "LAWSUIT" in s:
        return "SEC_ACTION"
    if "HACK" in s or "EXPLOIT" in s:
        return "HACK"
    if "FORK" in s or "HARD FORK" in s:
        return "FORK"
    return "OTHER"


def run_realtime_analysis(trigger: Trigger) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ_rt")
    slot = f"rt:{trigger.type.lower()}"

    # 0) Mark alert priority window and compute priority
    pr = _alert_prio(trigger.type, window_sec=int(os.getenv("RT_ALERT_WINDOW_SEC", "180")))

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
    # Extract last feature row for Alpha strategies match (best-effort)
    last_row = {}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ..infra.s3 import download_bytes as _dl
        buf = _dl(features_s3)
        df = pq.read_table(pa.BufferReader(buf)).to_pandas().sort_values("ts")
        if not df.empty:
            last_row = df.iloc[-1].to_dict()
    except Exception:
        last_row = {}

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
    # Trust weights (regime-based; optionally event-based for NEWS)
    trust = fetch_model_trust_regime(str(regime.get("label", "range")), "4h")  # fallback trust from 4h
    event_type = None
    if trigger.type == "NEWS":
        event_type = _classify_news_event(str(trigger.meta.get("title", "")))
        try:
            ev_trust = fetch_model_trust_regime_event(str(regime.get("label", "range")), "4h", event_type)
            if ev_trust:
                trust.update(ev_trust)
        except Exception:
            pass
    out = call_tool(
        "run_models_and_ensemble",
        {"features_s3": features_s3, "horizon": hz, "neighbors": topk, "trust_weights": trust},
    ) or {}
    if not out:
        try:
            from ..models.models import ModelsInput as _MIn, run as _run_models
            from ..ensemble.ensemble_rank import EnsembleInput as _EIn, run as _run_ens

            mm = _run_models(_MIn(features_path_s3=features_s3, horizon_minutes=hmin))
            preds = [p.model_dump() for p in mm.preds]
            ee = _run_ens(_EIn(preds=preds, horizon=hz, trust_weights=trust, neighbors=topk))
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
    # Apply probability calibration (isotonic + uncertainty)
    try:
        p_cal = calibrate_proba(
            float(e.get("proba_up", 0.5) or 0.5),
            (float((e.get("interval") or [0.0, 0.0])[0]), float((e.get("interval") or [0.0, 0.0])[1])),
            last_price,
            atr,
            hz,
        )
    except Exception:
        p_cal = float(e.get("proba_up", 0.5) or 0.5)
    upsert_prediction(
        run_id,
        hz,
        float(e.get("y_hat", 0.0) or 0.0),
        float((e.get("interval") or [0.0, 0.0])[0]),
        float((e.get("interval") or [0.0, 0.0])[1]),
        float(p_cal),
        {p.get("model"): {k: p.get(k) for k in ("y_hat", "pi_low", "pi_high", "proba_up", "cv_metrics")} for p in preds},
    )
    upsert_ensemble_weights(run_id, e.get("weights") or {})

    # 4) Chart reasoning (LLM-based technical view)
    try:
        ta = run_chart_reasoning(_CRInput(features_path_s3=features_s3))
    except Exception:
        ta = {"technical_sentiment": "neutral", "confidence_score": 0.5, "key_observations": []}

    # Event study for NEWS (optional)
    ev_summary = None
    if trigger.type == "NEWS" and event_type and event_type != "OTHER":
        try:
            ev_summary = call_tool("run_event_study", {"event_type": event_type, "k": 10, "window_hours": 24})
        except Exception:
            ev_summary = None

    # 4.5) Strategic agents context (latest verdicts)
    smc = None
    whale = None
    def _is_recent(ts_iso: str | None, win_sec: int) -> bool:
        if not ts_iso:
            return False
        try:
            from datetime import datetime, timezone
            t = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - t).total_seconds() <= win_sec
        except Exception:
            return False
    try:
        smc = fetch_latest_strategic_verdict("SMC Analyst", trigger.symbol)
    except Exception:
        smc = None
    try:
        whale = fetch_latest_strategic_verdict("Whale Watcher", trigger.symbol)
    except Exception:
        whale = None
    # Filter by alert window
    try:
        win = int(os.getenv("RT_ALERT_WINDOW_SEC", "180"))
        if smc and not _is_recent(smc.get("ts"), win):
            smc = None
        if whale and not _is_recent(whale.get("ts"), win):
            whale = None
    except Exception:
        pass

    # 5) Trade card
    tri = TradeRecommendInput(
        current_price=last_price,
        y_hat_4h=float(e.get("y_hat", 0.0) or 0.0),  # reuse field for short horizon
        interval_low=float((e.get("interval") or [0.0, 0.0])[0]),
        interval_high=float((e.get("interval") or [0.0, 0.0])[1]),
        proba_up=float(p_cal),
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

    # Confidence aggregation with factors
    alpha_match: Optional[AlphaMatchResult] = None
    if last_row:
        try:
            alpha_match = _alpha_match(last_row)
        except Exception:
            alpha_match = None
    try:
        ctx = {
            "trigger_type": trigger.type,
            "ta_sentiment": (ta or {}).get("technical_sentiment", "neutral"),
            "regime_label": str(regime.get("label", "range")),
            "side": card.get("side"),
            "pattern_hit": bool(trigger.type.startswith("PATTERN_")),
            "deriv_signal": bool(trigger.type.startswith("DERIV_")),
            "news_impact": float(trigger.meta.get("impact_score", 0.0) or 0.0) if trigger.type == "NEWS" else 0.0,
            "alert_priority": pr.label,
            "smc_status": (smc or {}).get("verdict", ""),
            "whale_status": (whale or {}).get("verdict", ""),
            "alpha_support": bool(alpha_match.matched) if alpha_match else False,
            "alpha_score": float(alpha_match.score) if alpha_match else None,
        }
        conf_res = compute_conf(float(card.get("confidence", 0.0) or 0.0), ctx)
        card["confidence"] = conf_res.value
        card["confidence_factors"] = conf_res.factors
        rc = card.get("reason_codes") or []
        rc.append("confidence:aggregated")
        card["reason_codes"] = rc
        if alpha_match and alpha_match.matched:
            try:
                push_values(
                    job="alpha_confidence",
                    values={
                        "alpha_score": float(alpha_match.score),
                        "final_confidence": float(conf_res.value),
                    },
                    labels={
                        "strategy": str(alpha_match.strategy_name or "unknown"),
                        "trigger": trigger.type,
                    },
                )
            except Exception:
                pass
    except Exception:
        pass

    # 6) Explain (short note)
    reason_text = _short_reason(trigger)
    md = (
        f"<b>Real-Time сигнал</b> ({hz})\n"
        f"Приоритет: <b>{pr.label.upper()}</b> (score={pr.score})\n"
        f"{reason_text}\n\n"
        + (f"Тех.сентимент: {ta.get('technical_sentiment','n/a')} (conf≈{float(ta.get('confidence_score',0.0)):.2f})\n" if ta else "")
        + (f"SMC: {smc.get('verdict')}\n" if smc and smc.get('verdict') else "")
        + (f"Whale: {whale.get('verdict')}\n" if whale and whale.get('verdict') else "")
        + (
            f"Alpha: {alpha_match.strategy_name} score≈{alpha_match.score:.2f}\n"
            if alpha_match and alpha_match.matched
            else ""
        )
        + f"p_up(cal)≈{float(p_cal):.2f}, ŷ≈{float(e.get('y_hat',0.0)):.2f}\n"
        + f"Рекомендация: {card.get('side')} lev×{card.get('leverage')} SL={card.get('stop_loss')} TP={card.get('take_profit')} conf≈{card.get('confidence')}\n"
        + (f"Событие: {event_type}\n" if event_type else "")
        + (f"EventStudy: n={ev_summary.get('n')} avg={ev_summary.get('avg_change'):.3f} p_pos={ev_summary.get('p_positive'):.2f}\n" if isinstance(ev_summary, dict) and ev_summary.get('n') else "")
        + f"Горизонт: ~{hmin} мин"
    )
    upsert_explanations(run_id, md, [])

    # Optional chart attachment (S3 URL)
    try:
        if os.getenv("RT_CHART_ENABLE", "1") in {"1", "true", "True"} and features_s3:
            # Lazy import to avoid hard dependency if matplotlib not installed
            from ..reporting.charts import plot_price_with_levels as _plot_price
            levels = []
            try:
                levels = [float(card.get("stop_loss")), float(card.get("take_profit"))]
            except Exception:
                levels = []
            chart_url = _plot_price(features_s3, title=f"{trigger.symbol} {hz}", y_hat_4h=float(e.get("y_hat", 0.0) or 0.0), levels=levels, slot=slot)
            if chart_url:
                md = md + f"\nГрафик: {chart_url}"
    except Exception:
        pass

    # Optional: publish concise Telegram alert with Redis-based dedup
    # Publish only if not NO-TRADE, verification passed, and not recently sent similar alert
    try:
        if card.get("side") != "NO-TRADE" and ok:
            # Dedup key: type+horizon+side bucket
            try:
                side_est = str(card.get("side") or ("LONG" if p_cal >= 0.5 else "SHORT"))
                dedup_sec = int(os.getenv("RT_ALERT_DEDUP_SEC", "120"))
                dkey = f"rt:dedup:{trigger.type}:{hz}:{side_est}"
                recent = _get_cache(dkey)
                if recent:
                    try:
                        push_values(job="rt_master", values={"dedup_skipped": 1.0}, labels={"type": trigger.type, "horizon": hz})
                    except Exception:
                        pass
                    return {
                        "run_id": run_id,
                        "slot": slot,
                        "horizon": hz,
                        "trigger": trigger.type,
                        "trade": {"card": card, "verified": ok, "reason": reason, "dedup": True},
                    }
                _set_cache(dkey, dedup_sec, "1")
            except Exception:
                pass
            rt_chat = os.getenv("TELEGRAM_RT_CHAT_ID")
            if rt_chat:
                publish_message_to(rt_chat, md)
            else:
                publish_message(md)
        else:
            try:
                push_values(job="rt_master", values={"ignored": 1.0}, labels={"type": trigger.type})
            except Exception:
                pass
    except Exception:
        pass

    # Metrics
    try:
        push_values(
            job="rt_master",
            values={
                "latency_sec": float(max(0, int(datetime.now(timezone.utc).timestamp()) - int(trigger.ts or 0))),
                "p_up": float(e.get("proba_up", 0.0) or 0.0),
            },
            labels={"type": trigger.type, "horizon": hz},
        )
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
    start_health_server()
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
