# flake8: noqa
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, cast

from loguru import logger

from ..data.ingest_news import IngestNewsInput
from ..data.ingest_news import run as run_news
from ..data.ingest_onchain import IngestOnchainInput
from ..data.ingest_onchain import run as run_onchain
from ..data.ingest_orderbook import IngestOrderbookInput
from ..data.ingest_orderbook import run as run_orderbook
from ..data.ingest_prices import IngestPricesInput
from ..data.ingest_prices import run as run_prices
from ..ensemble.ensemble_rank import EnsembleInput, EnsembleOutput
from ..ensemble.ensemble_rank import run as run_ensemble
from ..features.features_calc import FeaturesCalcInput
from ..features.features_calc import run as run_features
from ..infra.db import (
    log_error,
    upsert_ensemble_weights,
    upsert_explanations,
    upsert_features_snapshot,
    upsert_prediction,
    upsert_regime,
    upsert_scenarios,
    upsert_trade_suggestion,
    fetch_model_trust_regime,
)
from ..infra.metrics import push_durations, push_values, timed
from ..infra.obs import init_sentry
from ..infra.run_lock import acquire_release_lock
from ..models.models import ModelsInput, ModelsOutput
from ..models.models import run as run_models
from ..agents.memory_guardian_agent import MemoryGuardianAgent
from ..agents.investment_arbiter import InvestmentArbiter
from ..reasoning.debate_arbiter import debate
from ..agents.lessons import get_relevant_lessons
from ..reasoning.explain import explain_short
from ..regime.predictor_ml import predict as predict_regime_ml
from ..regime.regime_detect import Regime
from ..regime.regime_detect import detect as detect_regime
from ..reporting.charts import plot_price_with_levels
from ..reporting.release_report import save_release_report
from ..scenarios.scenario_helper import run_llm_or_fallback as run_scenarios_llm
from ..similarity.similar_past import SimilarPastInput
from ..similarity.similar_past import run as run_similar
from ..telegram_bot.publisher import (
    publish_code_block_json,
    publish_message,
    publish_photo_from_s3,
)
from ..trading.trade_recommend import TradeRecommendInput
from ..trading.trade_recommend import run as run_trade
from ..trading.verifier import verify
from ..utils.calibration import calibrate_proba
from ..utils.horizons import horizon_to_minutes


def predict_release(
    slot: str = "manual", horizon_hours_4h: int = 4, horizon_hours_12h: int = 12
):
    # Optional: delegate to AgentCoordinator-based DAG
    if os.getenv("USE_COORDINATOR", "0") in {"1", "true", "True"}:
        from .agent_flow import run_release_flow

        run_release_flow(
            slot=slot,
            horizon_hours_4h=horizon_hours_4h,
            horizon_hours_12h=horizon_hours_12h,
        )
        return
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")

    # Idempotency/once-per-slot guard (best-effort)
    acquired, lock_key = acquire_release_lock(slot)
    if not acquired:
        logger.info(f"Skip predict_release: lock exists for {lock_key}")
        return

    durations: dict[str, float] = {}
    # Ingest prices (last 72h for modeling)
    end_ts = int(now.timestamp())
    start_ts = end_ts - 72 * 3600
    p_in = IngestPricesInput(
        run_id=run_id, slot=slot, symbols=["BTCUSDT"], start_ts=start_ts, end_ts=end_ts
    )
    with timed(durations, "ingest_prices"):
        p_out = run_prices(p_in)

    # Ingest news (last 12h)
    n_in = IngestNewsInput(run_id=run_id, slot=slot, time_window_hours=12)
    with timed(durations, "ingest_news"):
        n_out = run_news(n_in)

    # Optional: Ingest orderbook snapshot
    ob_meta = None
    if os.getenv("ENABLE_ORDERBOOK", "0") in {"1", "true", "True"}:
        try:
            with timed(durations, "ingest_orderbook"):
                ob_out = run_orderbook(
                    IngestOrderbookInput(
                        run_id=run_id, slot=slot, symbol="BTCUSDT", depth=50
                    )
                )
                ob_meta = ob_out.meta
        except Exception:
            ob_meta = None

    # Optional: Ingest on-chain signals (placeholder)
    onchain_signals = []
    if os.getenv("ENABLE_ONCHAIN", "0") in {"1", "true", "True"}:
        try:
            with timed(durations, "ingest_onchain"):
                oc_out = run_onchain(
                    IngestOnchainInput(run_id=run_id, slot=slot, asset="BTC")
                )
                onchain_signals = [s.model_dump() for s in oc_out.onchain_signals]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"On-chain ingest failed: {exc}")
            onchain_signals = []

    # Features
    f_in = FeaturesCalcInput(
        prices_path_s3=p_out.prices_path_s3,
        news_signals=[s.model_dump() for s in n_out.news_signals],
        orderbook_meta=ob_meta,
        onchain_signals=onchain_signals,
        run_id=run_id,
        slot=slot,
    )
    with timed(durations, "features_calc"):
        f_out = run_features(f_in)
    # Persist features snapshot (best-effort)
    try:
        upsert_features_snapshot(run_id, f_out.snapshot_ts, f_out.features_path_s3)
    except Exception:
        pass

    # Regime detection (ML placeholder -> fallback heuristic)
    with timed(durations, "regime_detect"):
        try:
            rml = predict_regime_ml(f_out.features_path_s3)
            regime = Regime(
                label=rml.label,
                confidence=float(max(rml.proba.values() or [0.0])),
                features=rml.features,
            )
        except Exception:
            regime = detect_regime(f_out.features_path_s3)
    try:
        upsert_regime(run_id, regime.label, regime.confidence, regime.features)
    except Exception:
        pass

    # Similar past windows
    neighbors = []
    try:
        neighbors = [
            n.__dict__
            for n in run_similar(
                SimilarPastInput(
                    features_path_s3=f_out.features_path_s3, symbol="BTCUSDT", k=5
                )
            )
        ]
        from ..infra.db import upsert_similar_windows as _up_sw

        _up_sw(run_id, neighbors)
    except Exception:
        neighbors = []

    forecast_specs = [
        ("30m", 30),
        ("4h", horizon_hours_4h * 60),
        ("12h", horizon_hours_12h * 60),
        ("24h", 24 * 60),
    ]
    horizon_minutes_map = {hz: minutes for hz, minutes in forecast_specs}
    model_outputs: Dict[str, ModelsOutput] = {}
    ensemble_outputs: Dict[str, EnsembleOutput] = {}
    for hz, minutes in forecast_specs:
        with timed(durations, f"models_{hz}"):
            model_outputs[hz] = run_models(
                ModelsInput(
                    features_path_s3=f_out.features_path_s3,
                    horizon_minutes=minutes,
                )
            )
        with timed(durations, f"ensemble_{hz}"):
            ensemble_outputs[hz] = run_ensemble(
                EnsembleInput(
                    preds=[p.model_dump() for p in model_outputs[hz].preds],
                    horizon=hz,
                )
            )

    m4 = model_outputs["4h"]
    m12 = model_outputs["12h"]
    m30 = model_outputs.get("30m")
    m24 = model_outputs.get("24h")
    e4 = ensemble_outputs["4h"]
    e12 = ensemble_outputs["12h"]
    e30 = ensemble_outputs.get("30m")
    e24 = ensemble_outputs.get("24h")

    def _horizon_score(label: str, ensemble: EnsembleOutput | None) -> float:
        if not ensemble:
            return -1.0
        proba = ensemble.proba_up
        y_hat = ensemble.y_hat
        if y_hat >= 0 and proba >= 0.5:
            return proba - 0.5
        if y_hat < 0 and proba <= 0.5:
            return 0.5 - proba
        return -1.0

    primary_label = "4h"
    primary_ensemble = e4
    primary_models = m4
    best_score = _horizon_score("4h", e4)
    for label in ("30m", "12h"):
        ens = ensemble_outputs.get(label)
        score = _horizon_score(label, ens)
        if score > best_score + 0.05 and ens is not None:
            best_score = score
            primary_label = label
            primary_ensemble = ens
            primary_models = model_outputs.get(label, primary_models)

    primary_minutes = horizon_minutes_map.get(primary_label, horizon_hours_4h * 60)

    guardian_lessons: List[dict] = []
    guardian = MemoryGuardianAgent()
    try:
            mg = guardian.query(
                {
                    "scope": "trading",
                    "context": {
                        "planned_signal": "BULLISH" if primary_ensemble.y_hat >= 0 else "BEARISH",
                        "market_regime": regime.label,
                        "probability_up": primary_ensemble.proba_up,
                    },
                    "top_k": 3,
                }
            )
        guardian_lessons = (
            mg.output.get("lessons", []) if mg and isinstance(mg.output, dict) else []
        )
    except Exception:
        guardian_lessons = []

    arbiter = InvestmentArbiter()

    # Scenarios + chart
    with timed(durations, "scenarios"):
        scenarios, lvls = run_scenarios_llm(
            f_out.features_path_s3, current_price=m4.last_price, atr=m4.atr, slot=slot
        )
    with timed(durations, "chart_render"):
        chart_s3 = plot_price_with_levels(
            f_out.features_path_s3,
            title=f"BTC {slot} — 30m/4h/12h/24h прогноз",
            y_hat_4h=e4.y_hat,
            y_hat_12h=e12.y_hat,
            levels=lvls,
            slot=slot,
        )

    # Explain / Debate (LLM if доступен)
    news_points = [f"{s.title} ({s.source})" for s in n_out.news_signals[:3]]
    # Feedback loop: учесть релевантные уроки из недавнего прошлого
    lessons = []
    try:
        lessons = get_relevant_lessons({
            "regime": regime.label,
            "news": news_points,
            "ta": {"atr": primary_models.atr, "interval": primary_ensemble.interval},
        })
    except Exception:
        lessons = []

    # Optional RAG knowledge points
    rag_points = []
    try:
        import os
        if os.getenv("ENABLE_RAG", "0") in {"1", "true", "True"}:
            from ..knowledge.loader import search as _rag_search

            q = f"Regime:{regime.label}; News:{'; '.join(news_points[:3])}; Patterns:{'; '.join(primary_ensemble.rationale_points[:3])}"
            hits = _rag_search(q, k=5)
            rag_points = [h.get("title") or (h.get("content") or "")[:80] for h in hits]
    except Exception:
        rag_points = []

    deb_text, risk_flags = debate(
        rationale_points=primary_ensemble.rationale_points,
        regime=regime.label,
        news_top=news_points,
        neighbors=neighbors,
        lessons=guardian_lessons,
        knowledge=rag_points,
    )
    expl_text = explain_short(primary_ensemble.y_hat, primary_ensemble.proba_up, news_points, primary_ensemble.rationale_points)

    planned_side = "LONG" if primary_ensemble.y_hat >= 0 else "SHORT"
    safe_mode_active = os.getenv("SAFE_MODE", "0") in {"1", "true", "True"}
    trust_label = primary_label if primary_label in {"30m", "4h", "12h", "24h"} else "4h"
    try:
        model_trust_primary = fetch_model_trust_regime(regime.label, trust_label)
        if not model_trust_primary and trust_label != "4h":
            model_trust_primary = fetch_model_trust_regime(regime.label, "4h")
    except Exception:
        model_trust_primary = {}

    def _serialize(items):
        out = []
        for item in items or []:
            if isinstance(item, dict):
                out.append(item)
            elif hasattr(item, "model_dump"):
                out.append(item.model_dump())  # type: ignore[attr-defined]
            else:
                payload = {k: getattr(item, k) for k in dir(item) if not k.startswith("_") and not callable(getattr(item, k))}
                out.append(payload)
        return out

    context_payload = {
        "regime": {"label": regime.label, "confidence": regime.confidence, "features": regime.features},
        "news": _serialize(getattr(n_out, "news_signals", [])),
        "onchain": _serialize(onchain_signals),
        "neighbors": _serialize(neighbors),
        "lessons_context": {
            "scope": "trading",
            "regime": regime.label,
            "market_regime": regime.label,
            "planned_signal": "BULLISH" if planned_side == "LONG" else "BEARISH",
        },
        "smc": {},
        "advanced_ta": {},
        "features_path_s3": f_out.features_path_s3,
    }

    arbiter_out = arbiter.decide(
        {
            "run_id": run_id,
            "slot": slot,
            "symbol": "BTC/USDT",
            "planned_side": planned_side,
            "regime_label": regime.label,
            "lessons": guardian_lessons,
            "model_trust": model_trust_primary,
            "risk_flags": risk_flags,
            "safe_mode": safe_mode_active,
            "base_proba_up": float(primary_ensemble.proba_up),
            "context_payload": context_payload,
            "features_path_s3": f_out.features_path_s3,
        }
    )

    arbiter_decision = arbiter_out.evaluation
    arbiter_analysis = arbiter_out.analysis
    arbiter_critique = arbiter_out.critique
    context_bundle = arbiter_out.context

    # Trade recommendation (selected horizon)
    trade = run_trade(
        TradeRecommendInput(
            current_price=primary_models.last_price,
            y_hat_4h=primary_ensemble.y_hat,
            interval_low=primary_ensemble.interval[0],
            interval_high=primary_ensemble.interval[1],
            proba_up=arbiter_decision.proba_up,
            atr=float(primary_models.atr_map.get(primary_label, primary_models.atr) if isinstance(primary_models.atr_map, dict) else primary_models.atr),
            atr_map=primary_models.atr_map,
            account_equity=1000.0,
            risk_per_trade=0.005,
            leverage_cap=25.0,
            regime=regime.label,
            now_ts=int(now.timestamp()),
            tz_name=os.environ.get("TIMEZONE", "Asia/Jerusalem"),
            valid_for_minutes=90,
            horizon_minutes=primary_minutes,
            forecast_label=primary_label,
        )
    )
    ok, reason = verify(
        trade,
        current_price=primary_models.last_price,
        leverage_cap=25.0,
        interval=primary_ensemble.interval,
        atr=primary_models.atr,
    )

    # News-aware veto: if high-impact recent news contradicts direction, mark NO-TRADE
    try:
        from dateutil.parser import isoparse  # type: ignore

        horizon_check_s = 6 * 3600
        trade_is_long = trade.get("side") == "LONG"
        veto = False
        for s in n_out.news_signals[:5]:
            ts_str = getattr(s, "ts", None)
            imp = float(getattr(s, "impact_score", 0.0) or 0.0)
            sent = str(getattr(s, "sentiment", "")).lower()
            if imp < 0.8 or not ts_str:
                continue
            try:
                ts = isoparse(ts_str)
                if (now - ts).total_seconds() > horizon_check_s:
                    continue
            except Exception:
                continue
            if (sent == "positive" and not trade_is_long) or (
                sent == "negative" and trade_is_long
            ):
                veto = True
                break
        if veto:
            ok = False
            reason = "news_conflict"
    except Exception:
        pass

    if isinstance(trade.get("reason_codes"), list):
        trade["reason_codes"].append(f"arbiter:{arbiter_decision.risk_stance}")
    else:
        trade["reason_codes"] = [f"arbiter:{arbiter_decision.risk_stance}"]
    if arbiter_decision.confidence_floor is not None:
        try:
            trade["confidence"] = round(
                max(float(trade.get("confidence", 0.0)), arbiter_decision.confidence_floor),
                2,
            )
        except Exception:
            pass
    trade["arbiter"] = {
        "success_probability": round(arbiter_decision.success_probability, 3),
        "risk_stance": arbiter_decision.risk_stance,
        "notes": arbiter_decision.notes,
        "mode": arbiter_out.mode,
        "analysis": arbiter_analysis.raw if arbiter_analysis else None,
        "critique": arbiter_critique.raw if arbiter_critique else None,
        "context_tokens": context_bundle.get("tokens_estimate"),
    }

    msg = []
    msg.append(f"<b>BTC Forecast — {slot}</b>")
    msg.append(f"Run: <code>{run_id}</code>")
    msg.append("")
    e4_proba_cal = calibrate_proba(
        arbiter_decision.proba_up, e4.interval, m4.last_price, m4.atr, "4h"
    )
    e12_proba_cal = calibrate_proba(
        e12.proba_up, e12.interval, m4.last_price, m4.atr, "12h"
    )
    msg.append(
        f"<b>4h</b>: ŷ={e4.y_hat:.2f} (p_up={e4_proba_cal:.2f} cal, interval=({e4.interval[0]:.2f}..{e4.interval[1]:.2f}))"
    )
    msg.append(
        f"<b>12h</b>: ŷ={e12.y_hat:.2f} (p_up={e12_proba_cal:.2f} cal, interval=({e12.interval[0]:.2f}..{e12.interval[1]:.2f}))"
    )
    msg.append(
        f"<b>Arbiter</b>: stance={arbiter_decision.risk_stance} (success={arbiter_decision.success_probability:.2f})"
    )
    if arbiter_analysis:
        msg.append(
            f"<b>Аналитик</b>: {arbiter_analysis.scenario} @ {arbiter_analysis.probability_pct:.1f}% — {arbiter_analysis.explanation[:180]}"
        )
        if arbiter_analysis.contradictions:
            msg.append(
                "Противоречия: " + "; ".join(arbiter_analysis.contradictions[:3])
            )
    if arbiter_critique:
        msg.append(
            f"<b>Критика</b>: {arbiter_critique.recommendation} (Δ={arbiter_critique.probability_adjustment:+.1f} п.п.)"
        )
        if arbiter_critique.counterarguments:
            msg.append("• " + arbiter_critique.counterarguments[0])
    if e30:
        price_ref = getattr(m30, "last_price", m4.last_price)
        atr_ref = getattr(m30, "atr", m4.atr)
        e30_proba_cal = calibrate_proba(
            e30.proba_up,
            e30.interval,
            price_ref,
            atr_ref,
            "30m",
        )
        msg.append(
            f"<b>30m</b>: ŷ={e30.y_hat:.2f} (p_up={e30_proba_cal:.2f} cal, interval=({e30.interval[0]:.2f}..{e30.interval[1]:.2f}))"
        )
    if e24:
        price_ref = getattr(m24, "last_price", m4.last_price)
        atr_ref = getattr(m24, "atr", m4.atr)
        e24_proba_cal = calibrate_proba(
            e24.proba_up,
            e24.interval,
            price_ref,
            atr_ref,
            "24h",
        )
        msg.append(
            f"<b>24h</b>: ŷ={e24.y_hat:.2f} (p_up={e24_proba_cal:.2f} cal, interval=({e24.interval[0]:.2f}..{e24.interval[1]:.2f}))"
        )
    msg.append("")
    msg.append(
        f"<b>Режим рынка</b>: {regime.label} (conf={regime.confidence:.2f}, vol≈{regime.features.get('vol_pct', 0):.2f}%)"
    )
    if ob_meta:
        msg.append(
            f"Orderbook: imbalance={ob_meta.get('imbalance', 0.0):+.2f} (bid_vol={ob_meta.get('bid_vol', 0.0):.0f}, ask_vol={ob_meta.get('ask_vol', 0.0):.0f})"
        )
    if onchain_signals:
        msg.append(
            f"On‑chain: {len(onchain_signals)} сигнал(ов); пример: {onchain_signals[0]['metric']}={onchain_signals[0]['value']}"
        )
    msg.append("<b>Почему так (кратко)</b>:\n" + expl_text)
    msg.append("")
    msg.append("<b>Арбитраж аргументов</b>:\n" + deb_text)
    msg.append("")
    news_n = min(5, len(n_out.news_signals))
    if news_n:
        msg.append("<b>Топ новости:</b>")
        for s in n_out.news_signals[:news_n]:
            if getattr(s, "url", None):
                line = f'• [{s.sentiment} · {s.impact_score}] <a href="{s.url}">{s.title}</a> ({s.source})'
            else:
                line = f"• [{s.sentiment} · {s.impact_score}] {s.title} ({s.source})"
            msg.append(line)
    else:
        msg.append("Новостей за период мало или ключи API не заданы.")

    msg.append("")
    if neighbors:
        msg.append("<b>Похожие окна</b>:")
        for nb in neighbors[:3]:
            msg.append(f"• {nb['period']} (dist={nb['distance']:.3f})")
        msg.append("")
    msg.append("<b>Сценарии (5 веток):</b>")
    for sc in scenarios:
        msg.append(
            f"• {sc['if_level']}: {sc['then_path']} (p={sc['prob']:.2f}); inv: {sc['invalidation']}"
        )
    msg.append("")
    if trade.get("side") == "NO-TRADE" or not ok:
        msg.append("<b>Сделка</b>: NO-TRADE (" + (reason or "policy") + ")")
    else:
        msg.append(
            "<b>Сделка</b>: "
            f"{trade['side']} x{trade['leverage']} entry≈{trade['entry']['zone'][0]:.2f} SL={trade['stop_loss']:.2f} TP={trade['take_profit']:.2f} R:R={trade['rr_expected']:.2f}"
        )
    msg.append("")
    msg.append(
        f"Артефакты: prices → {p_out.prices_path_s3.split('/', 2)[-1]}, features → {f_out.features_path_s3.split('/', 2)[-1]}, chart → {chart_s3.split('/', 2)[-1]}"
    )
    msg.append("")
    msg.append(
        "<i>Отказ от ответственности: это не инвестсовет. Решения о сделках вы принимаете самостоятельно, учитывая риски.</i>"
    )

    # Publish chart then message
    with timed(durations, "publish"):
        publish_photo_from_s3(chart_s3, caption="btc_forecast_caption", slot=slot)
        publish_message("\n".join(msg))
    # отправим JSON карточки сделки отдельным сообщением (удобно копировать)
    publish_code_block_json("Карточка сделки (JSON)", trade)
    logger.info("Release post prepared and published")

    # Persist predictions and scenarios (best-effort)
    try:
        per_model_dict: Dict[str, Dict[str, Any]] = {}
        for hz, model_out in model_outputs.items():
            per_model_dict[hz] = {
                p.model: {
                    "y_hat": p.y_hat,
                    "pi_low": p.pi_low,
                    "pi_high": p.pi_high,
                    "proba_up": p.proba_up,
                    "cv": p.cv_metrics,
                }
                for p in model_out.preds
            }
        for hz, ens in ensemble_outputs.items():
            per_model_payload = per_model_dict.get(hz, {})
            upsert_prediction(
                run_id,
                hz,
                ens.y_hat,
                ens.interval[0],
                ens.interval[1],
                ens.proba_up,
                per_model_payload,
            )
        weights_payload = {}
        for hz, ens in ensemble_outputs.items():
            weights_payload.update({f"{hz}:{m}": float(v) for m, v in ens.weights.items()})
        upsert_ensemble_weights(run_id, weights_payload)
        upsert_scenarios(run_id, scenarios, chart_s3)
        # Save explanation/debate
        md = "<b>Почему так</b>\n" + expl_text + "\n\n<b>Арбитраж</b>\n" + deb_text
        upsert_explanations(run_id, md, risk_flags)
        upsert_trade_suggestion(run_id, trade)

        # Save release report to S3
        news_export = []
        for s in n_out.news_signals[:5]:
            news_export.append(
                {
                    "ts": getattr(s, "ts", None),
                    "title": getattr(s, "title", None),
                    "source": getattr(s, "source", None),
                    "sentiment": getattr(s, "sentiment", None),
                    "impact_score": getattr(s, "impact_score", None),
                    "url": getattr(s, "url", None),
                }
            )
        _ = save_release_report(
            run_id=run_id,
            slot=slot,
            regime={
                "label": regime.label,
                "confidence": regime.confidence,
                **regime.features,
            },
            neighbors=neighbors,
            ensembles={
                hz: {
                    "y_hat": ens.y_hat,
                    "proba_up": ens.proba_up,
                    "interval": ens.interval,
                    "weights": ens.weights,
                }
                for hz, ens in ensemble_outputs.items()
            },
            per_model=per_model_dict,
            scenarios=scenarios,
            trade_card=trade,
            news_top=news_export,
            artifacts={
                "prices": p_out.prices_path_s3,
                "features": f_out.features_path_s3,
                "chart": chart_s3,
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"DB persist skipped: {e}")
    push_durations(job="predict_release", durations=durations, labels={"slot": slot})
    # Per-model metrics
    try:
        per_model_vals = {}
        for hz, model_out in model_outputs.items():
            for p in model_out.preds:
                if not isinstance(p.cv_metrics, dict):
                    continue
                if "smape" in p.cv_metrics:
                    per_model_vals[f"smape_{hz}_{p.model}"] = float(p.cv_metrics["smape"])  # type: ignore[assignment]
                if "da" in p.cv_metrics:
                    per_model_vals[f"da_{hz}_{p.model}"] = float(p.cv_metrics["da"])  # type: ignore[assignment]
        if per_model_vals:
            push_values(
                job="predict_release", values=per_model_vals, labels={"slot": slot}
            )
    except Exception:
        pass
    # business metrics
    try:
        values = {
            "p_up_4h": float(e4.proba_up),
            "p_up_12h": float(e12.proba_up),
            "p_up_4h_cal": float(
                calibrate_proba(e4.proba_up, e4.interval, m4.last_price, m4.atr, "4h")
            ),
            "interval_width_pct_4h": float(
                (e4.interval[1] - e4.interval[0]) / max(1e-6, m4.last_price)
            ),
            "atr_pct": float(m4.atr / max(1e-6, m4.last_price)),
            "no_trade": 1.0 if (trade.get("side") == "NO-TRADE" or not ok) else 0.0,
            "rr_expected": float(trade.get("rr_expected") or 0.0),
            "leverage": float(trade.get("leverage") or 0.0),
            "neighbors_k": float(len(neighbors)),
        }
        if ob_meta and isinstance(ob_meta.get("imbalance"), (int, float)):
            values["ob_imbalance"] = float(cast(float, ob_meta.get("imbalance", 0.0)))
        # avg news confidence if available
        try:
            confs = []
            for s in n_out.news_signals:
                val = getattr(s, "confidence", None)
                if val is not None:
                    confs.append(float(val))
            if confs:
                values["news_conf_avg"] = float(sum(confs) / len(confs))
        except Exception:
            pass
        push_values(job="predict_release", values=values, labels={"slot": slot})
    except Exception:
        pass

    # Persist compact run summary for agent memory (best-effort)
    try:
        from ..infra.db import insert_run_summary as _ins_rs

        final_summary = {
            "slot": slot,
            "regime": str(regime.label),
            "regime_conf": float(regime.confidence),
            "e4": {
                "y_hat": float(e4.y_hat),
                "proba_up": float(e4.proba_up),
                "interval": [float(e4.interval[0]), float(e4.interval[1])],
            },
            "e12": {
                "y_hat": float(e12.y_hat),
                "proba_up": float(e12.proba_up),
                "interval": [float(e12.interval[0]), float(e12.interval[1])],
            },
            "risk_flags": risk_flags,
        }
        _ins_rs(run_id, final_summary, None)
    except Exception:
        pass


def main():
    init_sentry()
    slot = os.environ.get("SLOT", "manual")
    try:
        predict_release(slot)
    except Exception as e:  # noqa: BLE001
        # SLA fallback: publish minimal post without trade
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%SZ")
        try:
            log_error(
                run_id, None, {"error": str(e)}, regime=None, features_digest=None
            )
        except Exception:
            pass
        msg = []
        msg.append(f"<b>BTC Forecast — {slot}</b>")
        msg.append(f"Run: <code>{run_id}</code>")
        msg.append("")
        msg.append(
            "Технические неполадки — публикуем fallback без сделки. Следующий релиз по расписанию."
        )
        msg.append("")
        msg.append(
            "<i>Отказ от ответственности: это не инвестсовет. Решения о сделках вы принимаете самостоятельно, учитывая риски.</i>"
        )
        publish_message("\n".join(msg))


if __name__ == "__main__":
    main()
