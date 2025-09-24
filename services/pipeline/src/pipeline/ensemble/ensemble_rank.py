from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

from pydantic import BaseModel

from ..utils.horizons import horizon_to_timedelta


class EnsembleInput(BaseModel):
    preds: List[dict]
    trust_weights: dict | None = None  # optional per-model trust multipliers
    horizon: str | None = None  # optional (e.g., "4h" or "12h")
    # Optional: list of similar windows for context-aware weighting
    # Each item: {"period": iso-ts string, "distance": float}
    neighbors: List[Dict[str, Any]] | None = None


class EnsembleOutput(BaseModel):
    y_hat: float
    interval: Tuple[float, float]
    proba_up: float
    weights: dict
    rationale_points: List[str]


def _calculate_weight(pred: dict, trust_weights: dict | None) -> float:
    """Calculates the weight for a single model's prediction based on sMAPE and trust."""
    smape_val = (pred.get("cv_metrics") or {}).get("smape")
    if not isinstance(smape_val, (int, float)):
        return 0.0
    smape = float(smape_val)
    base_weight = 1.0 / max(1e-3, smape)

    trust_multiplier = 1.0
    if trust_weights and pred.get("model") in trust_weights:
        try:
            trust_multiplier = float(trust_weights[pred.get("model")])
        except (ValueError, TypeError):
            trust_multiplier = 1.0

    return base_weight * max(0.0, trust_multiplier)


def run(payload: EnsembleInput) -> EnsembleOutput:
    # Optional: context-aware bonus weights from similar windows performance
    similar_bonus: dict[str, float] = {}
    try:
        if payload.neighbors and payload.horizon:
            similar_bonus = _bonus_from_similar_windows(
                payload.neighbors, payload.horizon
            )
    except Exception:
        # best-effort; fall back silently
        similar_bonus = {}
    # Try stacking if enabled and horizon provided
    use_stacking = os.getenv("ENABLE_STACKING", "0") in {"1", "true", "True"}
    if use_stacking and payload.horizon:
        try:
            from .stacking import combine_with_weights, suggest_weights

            w_suggest, n = suggest_weights(payload.horizon)
            if w_suggest and n >= 30:
                y_hat, low, high, proba_up = combine_with_weights(
                    payload.preds, w_suggest
                )
                rationale = [
                    "Стэкинг включён",
                    "Веса (meta): "
                    + ", ".join(f"{m}={w:.2f}" for m, w in w_suggest.items()),
                    f"Samples: {n}",
                ]
                return EnsembleOutput(
                    y_hat=float(y_hat),
                    interval=(float(low), float(high)),
                    proba_up=float(proba_up),
                    weights=w_suggest,
                    rationale_points=rationale,
                )
        except Exception:
            # fallback to default below
            pass

    # Default: weight by inverse sMAPE (lower error -> higher weight), adjusted by optional trust
    weights = {}
    for p in payload.preds:
        m = p.get("model")
        base = _calculate_weight(p, payload.trust_weights)
        if m in similar_bonus:
            base *= max(0.0, 1.0 + float(similar_bonus.get(m, 0.0)))
        weights[m] = base
    total_weight = sum(weights.values())

    if total_weight == 0:
        # Fallback to uniform weights if all weights are zero
        weights = {p["model"]: 1.0 for p in payload.preds}
        total_weight = float(len(payload.preds))

    norm_w = {m: w / total_weight for m, w in weights.items()}

    def wavg(key: str) -> float:
        return sum((float(p[key]) * norm_w[p["model"]]) for p in payload.preds)

    y_hat = wavg("y_hat")
    low = wavg("pi_low")
    high = wavg("pi_high")
    proba_up = wavg("proba_up")

    # Build sMAPE string safely
    smape_parts = []
    for p in payload.preds:
        m = p.get("model", "?")
        s = (p.get("cv_metrics") or {}).get("smape")
        if isinstance(s, (int, float)):
            smape_parts.append(f"{m}={s:.2f}")
        else:
            smape_parts.append(f"{m}=n/a")

    rationale = [
        f"Веса ансамбля: {', '.join(f'{m}={w:.2f}' for m,w in norm_w.items())}",
        f"Качество (sMAPE): {', '.join(smape_parts)}",
    ]
    if payload.trust_weights:
        rationale.append(
            "Доверие: "
            + ", ".join(f"{k}={float(v):.2f}" for k, v in payload.trust_weights.items())
        )
    if similar_bonus:
        rationale.append(
            "Похожие окна: "
            + ", ".join(f"{k}:+{float(v):.2f}" for k, v in similar_bonus.items())
        )

    return EnsembleOutput(
        y_hat=float(y_hat),
        interval=(float(low), float(high)),
        proba_up=float(proba_up),
        weights=norm_w,
        rationale_points=rationale,
    )


def _bonus_from_similar_windows(
    neighbors: List[Dict[str, Any]], horizon: str
) -> Dict[str, float]:
    """Estimate per-model bonus multipliers from similar historical windows.

    Approach:
    - For each neighbor period, find the nearest run_id via features_snapshot.ts_window.
    - Load predictions.per_model_json for that run+horizon and y_true from
      prediction_outcomes if available (fallback to CCXT fetch).
    - Accumulate MSE per model; convert to normalized skill in [0..1].
    - Bonus = B * skill, where B is ENSEMBLE_SIMILAR_BONUS (default 0.2).
    """
    from ..infra.db import get_conn
    import os

    def _nearest_run_id(ts_iso: str) -> str | None:
        sql = (
            "SELECT run_id, ABS(EXTRACT(EPOCH FROM (ts_window - %s::timestamptz))) AS diff "
            "FROM features_snapshot ORDER BY diff ASC LIMIT 1"
        )
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (ts_iso,))
                row = cur.fetchone()
                return str(row[0]) if row else None

    def _per_model_and_created(run_id: str) -> tuple[dict, str | None]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT per_model_json, created_at FROM predictions WHERE run_id=%s AND horizon=%s",
                    (run_id, horizon),
                )
                row = cur.fetchone()
                if not row:
                    return {}, None
                per_model, created_at = row
                return (per_model or {}), (created_at.isoformat() if created_at else None)

    def _y_true_for(created_at_iso: str) -> float | None:
        # Prefer cache → outcomes table → fallback to CCXT fetch
        # Try Redis cache first
        try:
            import os as _os
            import json as _json
            import redis as _redis  # type: ignore

            r = _redis.Redis(host=_os.getenv("REDIS_HOST", "redis"), port=int(_os.getenv("REDIS_PORT", "6379")), db=0)
            key = f"ytrue:{horizon}:{created_at_iso}"
            raw = r.get(key)
            if raw is not None:
                try:
                    val = float(_json.loads(raw))
                    return val
                except Exception:
                    pass
        except Exception:
            pass
        # Outcomes table
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT y_true FROM prediction_outcomes WHERE created_at=%s AND horizon=%s ORDER BY created_at DESC LIMIT 1",
                    (created_at_iso, horizon),
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    try:
                        val = float(row[0])
                        # backfill cache
                        try:
                            import os as _os
                            import json as _json
                            import redis as _redis  # type: ignore

                            r = _redis.Redis(host=_os.getenv("REDIS_HOST", "redis"), port=int(_os.getenv("REDIS_PORT", "6379")), db=0)
                            ttl = int(_os.getenv("YTRUE_CACHE_TTL_SEC", "86400"))
                            r.setex(f"ytrue:{horizon}:{created_at_iso}", ttl, _json.dumps(val))
                        except Exception:
                            pass
                        return val
                    except Exception:
                        pass
        try:
            import ccxt  # type: ignore
            import pandas as pd
            ex = getattr(ccxt, os.getenv("CCXT_PROVIDER", "binance"))(
                {"enableRateLimit": True}
            )
            ts = pd.Timestamp(created_at_iso, tz="UTC").to_pydatetime()
            ahead = ts + horizon_to_timedelta(horizon)
            market = "BTC/USDT"
            ohlcv = ex.fetch_ohlcv(
                market,
                timeframe="1m",
                since=int((ahead.timestamp() - 60 * 5) * 1000),
                limit=10,
            )
            if not ohlcv:
                return None
            tgt = int(ahead.timestamp() * 1000)
            row = min(ohlcv, key=lambda r: abs(r[0] - tgt))
            val = float(row[4])
            # store to cache
            try:
                import os as _os
                import json as _json
                import redis as _redis  # type: ignore

                r = _redis.Redis(host=_os.getenv("REDIS_HOST", "redis"), port=int(_os.getenv("REDIS_PORT", "6379")), db=0)
                ttl = int(_os.getenv("YTRUE_CACHE_TTL_SEC", "86400"))
                r.setex(f"ytrue:{horizon}:{created_at_iso}", ttl, _json.dumps(val))
            except Exception:
                pass
            return val
        except Exception:
            return None

    # Accumulate squared errors per model across neighbors
    mse: Dict[str, float] = {}
    cnt: Dict[str, int] = {}
    for nb in neighbors[: max(1, int(os.getenv("SIMILAR_TOPK_LIMIT", "5")) )]:
        ts = str(nb.get("period") or nb.get("ts") or "")
        if not ts:
            continue
        rid = _nearest_run_id(ts)
        if not rid:
            continue
        per_model, created_iso = _per_model_and_created(rid)
        if not created_iso or not per_model:
            continue
        yt = _y_true_for(created_iso)
        if yt is None:
            continue
        for m, obj in (per_model or {}).items():
            try:
                yh = float((obj or {}).get("y_hat"))
            except Exception:
                continue
            e = (float(yt) - yh) ** 2
            mse[m] = mse.get(m, 0.0) + float(e)
            cnt[m] = cnt.get(m, 0) + 1

    if not mse:
        return {}
    # Convert to skill (lower MSE -> higher skill), normalize to [0..1]
    import math

    skill: Dict[str, float] = {}
    for m, s in mse.items():
        n = max(1, cnt.get(m, 1))
        rmse = math.sqrt(s / n)
        skill[m] = 1.0 / (1e-6 + rmse)
    if not skill:
        return {}
    smax = max(skill.values()) or 1.0
    norm = {m: (v / smax) for m, v in skill.items()}
    B = float(os.getenv("ENSEMBLE_SIMILAR_BONUS", "0.2"))
    return {m: B * max(0.0, min(1.0, v)) for m, v in norm.items()}
