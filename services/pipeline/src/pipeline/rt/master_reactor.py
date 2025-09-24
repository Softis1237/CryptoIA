from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
from ..infra.db import (
    fetch_model_trust_regime_event,
    fetch_model_trust_regime,
    fetch_agent_config,
)
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
from ..utils.horizons import (
    default_forecast_horizons,
    horizon_to_minutes,
    minutes_to_horizon,
)


@dataclass
class Trigger:
    type: str
    symbol: str
    ts: int
    meta: Dict[str, Any]


_RT_CONFIG_CACHE: Dict[str, Any] | None = None


def _rt_cfg() -> Dict[str, Any]:
    global _RT_CONFIG_CACHE
    if _RT_CONFIG_CACHE is None:
        data = fetch_agent_config("RealtimeMaster") or {}
        params = data.get("parameters") if isinstance(data, dict) else None
        _RT_CONFIG_CACHE = params or {}
    return _RT_CONFIG_CACHE


def _rt_list(key: str, env_key: str | None, default: List[str]) -> List[str]:
    cfg_val = _rt_cfg().get(key)
    if isinstance(cfg_val, (list, tuple)):
        items = [str(v).strip() for v in cfg_val if str(v).strip()]
        if items:
            return items
    if isinstance(cfg_val, str) and cfg_val.strip():
        items = [v.strip() for v in cfg_val.split(",") if v.strip()]
        if items:
            return items
    if env_key:
        env_val = os.getenv(env_key)
        if env_val:
            items = [v.strip() for v in env_val.split(",") if v.strip()]
            if items:
                return items
    return list(default)


def _rt_float(key: str, env_key: str | None, default: float) -> float:
    cfg_val = _rt_cfg().get(key)
    try:
        if cfg_val is not None:
            return float(cfg_val)
    except Exception:  # noqa: BLE001
        pass
    if env_key:
        env_val = os.getenv(env_key)
        if env_val:
            try:
                return float(env_val)
            except Exception:  # noqa: BLE001
                pass
    return default


def _rt_int(key: str, env_key: str | None, default: int) -> int:
    return int(round(_rt_float(key, env_key, float(default))))


PRICE_DEFAULT_PROVIDER = os.getenv("CCXT_PROVIDER", "binance")
PRICE_PROVIDERS = []
for prov in _rt_list("price_providers", "RT_PRICE_PROVIDERS", [PRICE_DEFAULT_PROVIDER]):
    if prov and prov not in PRICE_PROVIDERS:
        PRICE_PROVIDERS.append(prov)
if not PRICE_PROVIDERS:
    PRICE_PROVIDERS = [PRICE_DEFAULT_PROVIDER]

PRICE_TIMEOUT_MS = _rt_int("price_timeout_ms", "CCXT_TIMEOUT_MS", 20000)
PRICE_RETRIES = _rt_int("price_retries", "CCXT_RETRIES", 3)
PRICE_BACKOFF = _rt_float("price_backoff_sec", "CCXT_RETRY_BACKOFF_SEC", 1.0)


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
    lookback_sec = int(os.getenv("RT_PRICES_LOOKBACK_SEC", "172800"))
    price_errors: List[str] = []
    prices = None
    for provider in PRICE_PROVIDERS:
        try:
            prices = run_prices(
                IngestPricesInput(
                    run_id=run_id,
                    slot=slot,
                    symbols=[trigger.symbol.replace("/", "")],
                    start_ts=end_ts - lookback_sec,
                    end_ts=end_ts,
                    provider=provider,
                    fallback_providers=[p for p in PRICE_PROVIDERS if p != provider],
                    timeout_ms=PRICE_TIMEOUT_MS,
                    retries=PRICE_RETRIES,
                    retry_backoff_sec=PRICE_BACKOFF,
                )
            )
            break
        except Exception as exc:  # noqa: BLE001
            price_errors.append(f"{provider}:{exc}")
            try:
                push_values(
                    job="rt_master",
                    values={"price_ingest_fail": 1.0},
                    labels={"provider": provider},
                )
            except Exception:
                pass
            continue
    if prices is None:
        raise RuntimeError(f"Price ingestion failed: {'; '.join(price_errors) or 'no providers'}")
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
    primary_minutes = _horizon_for(trigger)
    primary_hz = minutes_to_horizon(primary_minutes)
    horizon_list = default_forecast_horizons()
    seen_hz: List[str] = []
    for hz_cfg in horizon_list:
        hz_norm = hz_cfg.lower().strip()
        if hz_norm not in seen_hz:
            seen_hz.append(hz_norm)
    if primary_hz not in seen_hz:
        seen_hz.insert(0, primary_hz)
    forecast_specs = []
    for hz_name in seen_hz:
        try:
            forecast_specs.append((hz_name, horizon_to_minutes(hz_name)))
        except ValueError:
            continue

    event_type = None
    if trigger.type == "NEWS":
        event_type = _classify_news_event(str(trigger.meta.get("title", "")))

    trust_map: Dict[str, Dict[str, float]] = {}
    regime_label = str(regime.get("label", "range"))
    for hz_name, _ in forecast_specs:
        trust_weights: Dict[str, float] = {}
        try:
            trust_weights = fetch_model_trust_regime(regime_label, hz_name) or {}
        except Exception:
            trust_weights = {}
        if trigger.type == "NEWS" and event_type:
            try:
                ev_trust = fetch_model_trust_regime_event(regime_label, hz_name, event_type)
                if ev_trust:
                    trust_weights.update(ev_trust)
            except Exception:
                pass
        trust_map[hz_name] = trust_weights

    ensemble_outputs: Dict[str, Dict[str, Any]] = {}
    model_info: Dict[str, Dict[str, Any]] = {}
    per_model_dict: Dict[str, Dict[str, Any]] = {}
    proba_cal_map: Dict[str, float] = {}
    weights_payload: Dict[str, float] = {}

    for hz_name, minutes in forecast_specs:
        trust_weights = trust_map.get(hz_name, {})
        out = call_tool(
            "run_models_and_ensemble",
            {
                "features_s3": features_s3,
                "horizon": hz_name,
                "neighbors": topk,
                "trust_weights": trust_weights,
            },
        ) or {}
        if not out:
            try:
                from ..models.models import ModelsInput as _MIn, run as _run_models
                from ..ensemble.ensemble_rank import EnsembleInput as _EIn, run as _run_ens

                mm = _run_models(
                    _MIn(features_path_s3=features_s3, horizon_minutes=minutes)
                )
                preds = [p.model_dump() for p in mm.preds]
                ee = _run_ens(
                    _EIn(
                        preds=preds,
                        horizon=hz_name,
                        trust_weights=trust_weights,
                        neighbors=topk,
                    )
                )
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

        ens = (out.get("ensemble") or {})
        preds = out.get("preds") or []
        last_price = float(out.get("last_price", 0.0) or 0.0)
        atr = float(out.get("atr", 0.0) or 0.0)
        ensemble_outputs[hz_name] = ens
        model_info[hz_name] = {
            "preds": preds,
            "last_price": last_price,
            "atr": atr,
        }
        per_model = {
            str(p.get("model")): {
                "y_hat": p.get("y_hat"),
                "pi_low": p.get("pi_low"),
                "pi_high": p.get("pi_high"),
                "proba_up": p.get("proba_up"),
                "cv_metrics": p.get("cv_metrics"),
            }
            for p in preds
            if p.get("model")
        }
        per_model_dict[hz_name] = per_model
        try:
            p_cal = calibrate_proba(
                float(ens.get("proba_up", 0.5) or 0.5),
                (
                    float((ens.get("interval") or [0.0, 0.0])[0]),
                    float((ens.get("interval") or [0.0, 0.0])[1]),
                ),
                last_price,
                atr,
                hz_name,
            )
        except Exception:
            p_cal = float(ens.get("proba_up", 0.5) or 0.5)
        proba_cal_map[hz_name] = p_cal
        upsert_prediction(
            run_id,
            hz_name,
            float(ens.get("y_hat", 0.0) or 0.0),
            float((ens.get("interval") or [0.0, 0.0])[0]),
            float((ens.get("interval") or [0.0, 0.0])[1]),
            float(p_cal),
            per_model,
        )
        weights_payload.update(
            {f"{hz_name}:{m}": float(w) for m, w in (ens.get("weights") or {}).items()}
        )

    if weights_payload:
        upsert_ensemble_weights(run_id, weights_payload)

    if primary_hz not in ensemble_outputs and forecast_specs:
        primary_hz = forecast_specs[0][0]
        primary_minutes = forecast_specs[0][1]
    else:
        primary_minutes = horizon_to_minutes(primary_hz)

    ens_primary = ensemble_outputs.get(primary_hz, {})
    model_primary = model_info.get(primary_hz, {})
    preds = model_primary.get("preds", [])
    last_price = float(model_primary.get("last_price", 0.0) or 0.0)
    atr = float(model_primary.get("atr", 0.0) or 0.0)
    p_cal = proba_cal_map.get(primary_hz, float(ens_primary.get("proba_up", 0.5) or 0.5))

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
        y_hat_4h=float(ens_primary.get("y_hat", 0.0) or 0.0),  # reuse field for short horizon
        interval_low=float((ens_primary.get("interval") or [0.0, 0.0])[0]),
        interval_high=float((ens_primary.get("interval") or [0.0, 0.0])[1]),
        proba_up=float(p_cal),
        atr=atr,
        regime=str(regime.get("label")),
        account_equity=float(os.getenv("ACCOUNT_EQUITY", "1000")),
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.005")),
        leverage_cap=float(os.getenv("RISK_LEVERAGE_CAP", "25")),
        now_ts=int(now.timestamp()),
        tz_name=os.getenv("TIMEZONE", "Europe/Moscow"),
        valid_for_minutes=int(os.getenv("RT_VALID_FOR_MIN", "30")),
        horizon_minutes=primary_minutes,
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
        f"<b>Real-Time сигнал</b> ({primary_hz})\n"
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
        + f"p_up(cal)≈{float(p_cal):.2f}, ŷ≈{float(ens_primary.get('y_hat',0.0)):.2f}\n"
        + f"Рекомендация: {card.get('side')} lev×{card.get('leverage')} SL={card.get('stop_loss')} TP={card.get('take_profit')} conf≈{card.get('confidence')}\n"
        + (f"Событие: {event_type}\n" if event_type else "")
        + (f"EventStudy: n={ev_summary.get('n')} avg={ev_summary.get('avg_change'):.3f} p_pos={ev_summary.get('p_positive'):.2f}\n" if isinstance(ev_summary, dict) and ev_summary.get('n') else "")
        + f"Горизонт: ~{primary_minutes} мин"
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
            chart_url = _plot_price(features_s3, title=f"{trigger.symbol} {primary_hz}", y_hat_4h=float(ens_primary.get("y_hat", 0.0) or 0.0), levels=levels, slot=slot)
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
                dkey = f"rt:dedup:{trigger.type}:{primary_hz}:{side_est}"
                recent = _get_cache(dkey)
                if recent:
                    try:
                        push_values(job="rt_master", values={"dedup_skipped": 1.0}, labels={"type": trigger.type, "horizon": primary_hz})
                    except Exception:
                        pass
                    return {
                        "run_id": run_id,
                        "slot": slot,
                        "horizon": primary_hz,
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
                "p_up": float(ens_primary.get("proba_up", 0.0) or 0.0),
            },
            labels={"type": trigger.type, "horizon": primary_hz},
        )
    except Exception:
        pass
    return {
        "run_id": run_id,
        "slot": slot,
        "horizon": primary_hz,
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
