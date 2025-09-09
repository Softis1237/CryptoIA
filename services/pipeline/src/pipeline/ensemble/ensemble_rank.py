from __future__ import annotations

from typing import List, Tuple

import os
from pydantic import BaseModel


class EnsembleInput(BaseModel):
    preds: List[dict]
    trust_weights: dict | None = None  # optional per-model trust multipliers
    horizon: str | None = None  # optional (e.g., "4h" or "12h")


class EnsembleOutput(BaseModel):
    y_hat: float
    interval: Tuple[float, float]
    proba_up: float
    weights: dict
    rationale_points: List[str]


def run(payload: EnsembleInput) -> EnsembleOutput:
    # Try stacking if enabled and horizon provided
    use_stacking = os.getenv("ENABLE_STACKING", "0") in {"1", "true", "True"}
    if use_stacking and payload.horizon:
        try:
            from .stacking import suggest_weights, combine_with_weights

            w_suggest, n = suggest_weights(payload.horizon)
            if w_suggest and n >= 30:
                y_hat, low, high, proba_up = combine_with_weights(payload.preds, w_suggest)
                rationale = [
                    "Стэкинг включён",
                    "Веса (meta): " + ", ".join(f"{m}={w:.2f}" for m, w in w_suggest.items()),
                    f"Samples: {n}",
                ]
                return EnsembleOutput(y_hat=float(y_hat), interval=(float(low), float(high)), proba_up=float(proba_up), weights=w_suggest, rationale_points=rationale)
        except Exception:
            # fallback to default below
            pass
    # weight by inverse sMAPE (lower error -> higher weight), adjusted by optional trust
    weights = {}
    total = 0.0
    for p in payload.preds:
        smape = float((p.get("cv_metrics") or {}).get("smape", 20.0))
        base = 1.0 / max(1e-3, smape)
        tw = 1.0
        if payload.trust_weights and p.get("model") in payload.trust_weights:
            try:
                tw = float(payload.trust_weights[p.get("model")])
            except Exception:
                tw = 1.0
        w = base * max(0.0, tw)
        weights[p["model"]] = w
        total += w
    if total == 0:
        weights = {p["model"]: 1.0 for p in payload.preds}
        total = float(len(payload.preds))

    norm_w = {m: w / total for m, w in weights.items()}

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
        rationale.append("Доверие: " + ", ".join(f"{k}={float(v):.2f}" for k,v in payload.trust_weights.items()))

    return EnsembleOutput(y_hat=float(y_hat), interval=(float(low), float(high)), proba_up=float(proba_up), weights=norm_w, rationale_points=rationale)
