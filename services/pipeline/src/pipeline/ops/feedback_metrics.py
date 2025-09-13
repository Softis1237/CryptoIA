from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..infra.db import (
    get_conn,
    upsert_prediction_outcome,
    upsert_model_trust_regime_event,
)
from ..infra.metrics import push_values


# --- Data helpers -----------------------------------------------------------


def _get_features_path(run_id: str) -> Optional[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT path_s3 FROM features_snapshot WHERE run_id=%s",
                (run_id,),
            )
            row = cur.fetchone()
            return str(row[0]) if row and row[0] else None


def _load_features_tail(path_s3: str) -> Optional[pd.Series]:
    try:
        from ..infra.s3 import download_bytes
        import pyarrow as pa
        import pyarrow.parquet as pq

        raw = download_bytes(path_s3)
        df = pq.read_table(pa.BufferReader(raw)).to_pandas()
        df = df.sort_values("ts")
        return df.tail(1).iloc[0]
    except Exception:
        return None


def _get_regime_label(run_id: str) -> Optional[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT label FROM regimes WHERE run_id=%s", (run_id,))
            row = cur.fetchone()
            return str(row[0]) if row and row[0] else None


def _get_agent_result(run_id: str, agent: str) -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT result_json FROM agents_predictions WHERE run_id=%s AND agent=%s",
                (run_id, agent),
            )
            row = cur.fetchone()
            try:
                return row[0] or {}
            except Exception:
                return {}


def _get_y_true(ts_iso: str, horizon: str, provider: str) -> Optional[float]:
    try:
        from ..ensemble.stacking import _get_y_true_ccxt as _yt
        return _yt(ts_iso, horizon, provider)
    except Exception:
        # Fallback: inline light version to avoid import issue
        try:
            import ccxt  # type: ignore
            ex = getattr(ccxt, provider)({"enableRateLimit": True})
            ts = pd.Timestamp(ts_iso, tz="UTC").to_pydatetime()
            ahead = ts + (timedelta(hours=4) if horizon == "4h" else timedelta(hours=12))
            market = "BTC/USDT"
            ohlcv = ex.fetch_ohlcv(market, timeframe="1m", since=int((ahead.timestamp() - 60 * 5) * 1000), limit=10)
            if not ohlcv:
                return None
            tgt = int(ahead.timestamp() * 1000)
            row = min(ohlcv, key=lambda r: abs(r[0] - tgt))
            return float(row[4])
        except Exception:
            return None


# --- Core feedback functions -----------------------------------------------


@dataclass
class Outcome:
    run_id: str
    horizon: str
    created_at_iso: str
    y_hat: float
    y_true: Optional[float]
    last_price: Optional[float]
    error_abs: Optional[float]
    error_pct: Optional[float]
    direction_correct: Optional[bool]
    regime_label: Optional[str]
    news_ctx: Optional[float]
    tags: List[str]


def tag_root_causes(
    *,
    error_abs: float | None,
    error_pct: float | None,
    last_price: float | None,
    news_ctx: float | None,
    regime_label: str | None,
    volume_z: float | None,
    macro_level: str | None,
    event_hypothesis: list | None,
    direction_correct: bool | None,
) -> List[str]:
    tags: List[str] = []
    if error_pct is None or last_price is None:
        return tags
    e = float(abs(error_pct))
    # Heuristics
    if regime_label in {"trend_up", "trend_down"} and direction_correct is False and e > 0.01:
        tags.append("regime_mismatch")
    if (news_ctx or 0.0) > 0.7 and direction_correct is False and e > 0.01:
        tags.append("false_news")
    if (volume_z or 0.0) > 2.0 and e > 0.008:
        tags.append("unexpected_volume_spike")
    if (macro_level or "none") in {"high", "medium"} and e > 0.012:
        tags.append("macro_surprise")
    if event_hypothesis and len(event_hypothesis) > 0 and e > 0.01:
        tags.append("event_impact_miss")
    # Always clip to unique
    return sorted(list(set(tags)))


def persist_outcome(o: Outcome) -> None:
    try:
        upsert_prediction_outcome(
            run_id=o.run_id,
            horizon=o.horizon,
            created_at_iso=o.created_at_iso,
            y_hat=float(o.y_hat) if o.y_hat is not None else None,
            y_true=float(o.y_true) if o.y_true is not None else None,
            error_abs=float(o.error_abs) if o.error_abs is not None else None,
            error_pct=float(o.error_pct) if o.error_pct is not None else None,
            direction_correct=bool(o.direction_correct) if o.direction_correct is not None else None,
            regime_label=o.regime_label,
            news_ctx=float(o.news_ctx) if o.news_ctx is not None else None,
            tags={"root_causes": o.tags},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"persist_outcome failed for run={o.run_id} hz={o.horizon}: {e}")


def collect_outcomes(horizon: str, provider: Optional[str] = None) -> Dict[str, float]:
    """Collect realized outcomes for matured predictions and publish metrics.

    - Computes y_true via CCXT based on prediction created_at and horizon.
    - Derives context: last_price, news_ctx_score, volume_z from features snapshot;
      regime_label from regimes; macro/event from agents_predictions.
    - Persists to prediction_outcomes and pushes aggregate metrics.
    """
    from ..infra.db import fetch_predictions_for_cv
    provider = provider or os.getenv("CCXT_PROVIDER", "binance")

    rows = fetch_predictions_for_cv(horizon, min_age_hours=1.0 if horizon == "1h" else (4.0 if horizon == "4h" else 12.0))
    outcomes: List[Outcome] = []
    for run_id, hz, created_at, y_hat, pi_low, pi_high, per_model in rows:
        y_true = _get_y_true(created_at, hz, provider)
        if y_true is None:
            continue
        # Context
        regime_label = _get_regime_label(run_id)
        fpath = _get_features_path(run_id)
        last_price = None
        news_ctx = None
        volume_z = None
        if fpath:
            tail = _load_features_tail(fpath)
            try:
                last_price = float(tail["close"]) if tail is not None else None
            except Exception:
                last_price = None
            try:
                news_ctx = float(tail.get("news_ctx_score", np.nan)) if tail is not None else None
                if news_ctx is not None and not (news_ctx == news_ctx):  # NaN
                    news_ctx = None
            except Exception:
                news_ctx = None
            try:
                volume_z = float(tail.get("volume_z", np.nan)) if tail is not None else None
                if volume_z is not None and not (volume_z == volume_z):
                    volume_z = None
            except Exception:
                volume_z = None
        macro = _get_agent_result(run_id, "macro_analyzer") or {}
        event = _get_agent_result(run_id, "event_impact") or {}
        macro_level = str(macro.get("level") or macro.get("macro_level") or "none")
        event_hyp = event.get("hypothesis") or event.get("event") or []

        # Metrics
        err_abs = abs(float(y_true) - float(y_hat or 0.0))
        denom = max(1e-9, abs(float(y_true)))
        err_pct = err_abs / denom
        if last_price is not None:
            direction_correct = np.sign(float(y_true) - float(last_price)) == np.sign(float(y_hat or 0.0) - float(last_price))
        else:
            direction_correct = None

        tags = tag_root_causes(
            error_abs=err_abs,
            error_pct=err_pct,
            last_price=last_price,
            news_ctx=news_ctx,
            regime_label=regime_label,
            volume_z=volume_z,
            macro_level=macro_level,
            event_hypothesis=event_hyp if isinstance(event_hyp, list) else [event_hyp],
            direction_correct=direction_correct,
        )
        outcomes.append(
            Outcome(
                run_id=run_id,
                horizon=hz,
                created_at_iso=created_at,
                y_hat=float(y_hat or 0.0),
                y_true=float(y_true),
                last_price=last_price,
                error_abs=float(err_abs),
                error_pct=float(err_pct),
                direction_correct=bool(direction_correct) if direction_correct is not None else None,
                regime_label=regime_label,
                news_ctx=news_ctx,
                tags=tags,
            )
        )

    # Persist
    for o in outcomes:
        persist_outcome(o)

    # Aggregates and metrics
    vals: Dict[str, float] = {}
    if outcomes:
        mae = float(np.mean([o.error_abs for o in outcomes if o.error_abs is not None]))
        smape = float(
            np.mean([
                (abs(o.y_true - o.y_hat) / (abs(o.y_true) + abs(o.y_hat) + 1e-9))
                for o in outcomes
                if o.y_true is not None and o.y_hat is not None
            ])
        )
        da_list = [
            1.0
            if (o.direction_correct is True)
            else 0.0
            for o in outcomes
            if o.direction_correct is not None
        ]
        da = float(np.mean(da_list)) if da_list else 0.0
        vals[f"mae_{horizon}"] = mae
        vals[f"smape_{horizon}"] = smape
        vals[f"da_{horizon}"] = da
        # By-regime sMAPE
        by_regime: Dict[str, List[float]] = defaultdict(list)
        for o in outcomes:
            if o.regime_label:
                by_regime[str(o.regime_label)].append(abs(o.y_true - o.y_hat) / (abs(o.y_true) + abs(o.y_hat) + 1e-9))
        for reg, arr in by_regime.items():
            if arr:
                vals[f"err_by_regime{horizon}_{reg}"] = float(np.mean(arr))
    if vals:
        try:
            from ..infra.db import insert_agent_metric
            for k, v in vals.items():
                insert_agent_metric("outcomes", k, float(v), labels={"horizon": horizon})
        except Exception:
            pass
        push_values(job="outcomes", values=vals, labels={})
        logger.info(f"collect_outcomes[{horizon}]: pushed {vals}")
    else:
        logger.info(f"collect_outcomes[{horizon}]: nothing to persist")
    return vals


def update_regime_alphas(window_days: int = None) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Compute per-regime per-model alpha weights using EWMA of sMAPE and persist.

    Returns a mapping {(regime, horizon) -> {model: alpha}} for observability.
    """
    window_days = window_days or int(os.getenv("REGIME_ALPHA_DECAY_DAYS", "14"))
    alpha_min = float(os.getenv("REGIME_ALPHA_MIN", "0.7"))
    alpha_max = float(os.getenv("REGIME_ALPHA_MAX", "1.3"))
    beta = float(os.getenv("REGIME_ALPHA_SMOOTH_BETA", "0.3"))  # EMA blend with previous

    # Load joined predictions + outcomes for window
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.run_id, p.horizon, p.created_at, p.per_model_json, o.y_true, o.regime_label
                FROM predictions p
                JOIN prediction_outcomes o USING (run_id, horizon)
                WHERE p.created_at >= %s
                """,
                (since,),
            )
            rows = cur.fetchall() or []

    # Organize by (regime, horizon)
    buckets: Dict[Tuple[str, str], List[Tuple[datetime, dict, float]]] = defaultdict(list)
    for run_id, hz, created_at, per_model, y_true, regime_label in rows:
        try:
            ts = created_at if isinstance(created_at, datetime) else datetime.fromisoformat(str(created_at))
        except Exception:
            ts = now
        reg = str(regime_label or "")
        if not reg:
            continue
        if not per_model or y_true is None:
            continue
        buckets[(reg, str(hz))].append((ts, per_model or {}, float(y_true)))

    # Also load event types per run_id (from agents_predictions)
    evmap: Dict[str, str] = {}
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT run_id, agent, result_json FROM agents_predictions WHERE agent IN ('event_impact') OR agent LIKE 'event_study_%' AND created_at >= %s",
                    (since,),
                )
                for rid, agent, res in cur.fetchall() or []:
                    et = None
                    try:
                        if agent == "event_impact":
                            hyp = (res or {}).get("hypothesis") or []
                            for h in hyp:
                                e = str(h.get("event", "")).upper()
                                if e == "ETF":
                                    et = "ETF_APPROVAL"
                                elif e == "REG":
                                    et = "SEC_ACTION"
                                elif e in {"HACK", "FORK"}:
                                    et = e
                                if et:
                                    break
                        elif agent.startswith("event_study_"):
                            et = agent.replace("event_study_", "").upper()
                    except Exception:
                        et = None
                    if et:
                        evmap[str(rid)] = et
    except Exception:
        evmap = {}

    def ewma_smape(samples: List[Tuple[datetime, dict, float]]) -> Dict[str, float]:
        # Compute EWMA sMAPE per model
        out: Dict[str, float] = {}
        weights_sum: Dict[str, float] = defaultdict(float)
        sums: Dict[str, float] = defaultdict(float)
        for ts, per_model, y_true in samples:
            age_days = max(0.0, (now - (ts if isinstance(ts, datetime) else now)).total_seconds() / 86400.0)
            w = math.exp(-age_days / max(1e-6, float(window_days)))
            for m, info in (per_model or {}).items():
                try:
                    yhat = float((info or {}).get("y_hat", 0.0))
                    sm = abs(yhat - y_true) / (abs(yhat) + abs(y_true) + 1e-9)
                except Exception:
                    sm = 1.0
                sums[m] += w * sm
                weights_sum[m] += w
        for m in sums.keys():
            if weights_sum[m] > 0:
                out[m] = float(sums[m] / weights_sum[m])
        return out

    result: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, samples in buckets.items():
        reg, hz = key
        smape_map = ewma_smape(samples)
        if not smape_map:
            continue
        # Convert to alpha via inverse error, normalize around mean=1, clip
        vals = {m: (1.0 / max(1e-6, v)) for m, v in smape_map.items()}
        s = sum(vals.values()) or 1.0
        norm = {m: v / (s / max(1, len(vals))) for m, v in vals.items()}  # mean ~1
        # Smooth with previous
        prev: Dict[str, float] = {}
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT model, weight FROM model_trust_regime WHERE regime_label=%s AND horizon=%s",
                        (reg, hz),
                    )
                    for m, w in cur.fetchall() or []:
                        prev[str(m)] = float(w or 1.0)
        except Exception:
            prev = {}
        alphas: Dict[str, float] = {}
        for m, v in norm.items():
            base = float(v)
            pr = float(prev.get(m, 1.0))
            a = pr * (1.0 - beta) + base * beta
            a = max(alpha_min, min(alpha_max, a))
            alphas[m] = float(a)
        # Persist
        from ..infra.db import upsert_model_trust_regime
        for m, a in alphas.items():
            try:
                upsert_model_trust_regime(reg, hz, m, float(a))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"alpha upsert failed: reg={reg} hz={hz} model={m}: {e}")
        result[key] = alphas

        # Event-type specific alphas (best-effort)
        try:
            # Reload rows for this (reg, hz) to get run_id
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT p.run_id, p.created_at, p.per_model_json, o.y_true
                        FROM predictions p
                        JOIN prediction_outcomes o USING (run_id, horizon)
                        WHERE o.regime_label=%s AND p.horizon=%s AND p.created_at >= %s
                        """,
                        (reg, hz, since),
                    )
                    r2 = cur.fetchall() or []
            # Group by event type present in evmap
            ev_buckets: Dict[str, List[Tuple[datetime, dict, float]]] = defaultdict(list)
            for rid, created_at, per_model, y_true in r2:
                et = evmap.get(str(rid))
                if not et:
                    continue
                try:
                    ts2 = created_at if isinstance(created_at, datetime) else datetime.fromisoformat(str(created_at))
                except Exception:
                    ts2 = now
                if per_model and (y_true is not None):
                    ev_buckets[str(et)].append((ts2, per_model or {}, float(y_true)))
            for et, s_list in ev_buckets.items():
                smape_ev = ewma_smape(s_list)
                if not smape_ev:
                    continue
                vals_ev = {m: (1.0 / max(1e-6, v)) for m, v in smape_ev.items()}
                s_ev = sum(vals_ev.values()) or 1.0
                norm_ev = {m: v / (s_ev / max(1, len(vals_ev))) for m, v in vals_ev.items()}
                # Smooth with previous event-specific weights
                prev_ev: Dict[str, float] = {}
                try:
                    with get_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT model, weight FROM model_trust_regime_event WHERE regime_label=%s AND horizon=%s AND event_type=%s",
                                (reg, hz, et),
                            )
                            for m, w in cur.fetchall() or []:
                                prev_ev[str(m)] = float(w or 1.0)
                except Exception:
                    prev_ev = {}
                for m, v in norm_ev.items():
                    base = float(v)
                    pr = float(prev_ev.get(m, 1.0))
                    a = pr * (1.0 - beta) + base * beta
                    a = max(alpha_min, min(alpha_max, a))
                    try:
                        upsert_model_trust_regime_event(reg, hz, et, m, float(a))
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"alpha(event) upsert failed: reg={reg} hz={hz} et={et} model={m}: {e}")
        except Exception:
            logger.exception("update_regime_alphas: event-specific alphas failed")

    # Export a snapshot for observability
    flat_vals: Dict[str, float] = {}
    for (reg, hz), amap in result.items():
        for m, a in amap.items():
            flat_vals[f"alpha_{hz}_{reg}_{m}"] = float(a)
    if flat_vals:
        push_values(job="regime_alpha", values=flat_vals, labels={})
    return result


# Legacy: keep simple feedback counters (UI feedback)
def _run_feedback_summary(window_days: int = 7) -> dict:
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT rating FROM users_feedback WHERE created_at >= %s AND rating IS NOT NULL",
                (since,),
            )
            rows = cur.fetchall() or []
            ratings = [int(r[0]) for r in rows if r and r[0] is not None]
    if ratings:
        avg = float(np.mean(ratings))
        cnt = len(ratings)
    else:
        avg = 0.0
        cnt = 0
    vals = {"feedback_avg": avg, "feedback_count": float(cnt)}
    push_values(job="feedback", values=vals, labels={})
    logger.info(f"feedback_metrics: avg={avg:.2f} count={cnt}")
    return {"avg": avg, "count": cnt}


def main():
    # Run both horizons best-effort
    try:
        collect_outcomes("4h")
        collect_outcomes("12h")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"collect_outcomes failed: {e}")
    try:
        update_regime_alphas()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"update_regime_alphas failed: {e}")
    try:
        _run_feedback_summary()
    except Exception:
        pass


if __name__ == "__main__":
    main()
