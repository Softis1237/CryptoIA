# flake8: noqa
from __future__ import annotations

# mypy: ignore-errors

"""Agent-based orchestration of the release flow (P1).

This module builds a DAG of lightweight agents wrapping existing pipeline steps
and executes them via AgentCoordinator. It aims to be a drop-in alternative to
the monolithic predict_release while keeping identical outputs and side effects
where possible.

Usage:
  python -m pipeline.orchestration.agent_flow --slot=manual

Env flags:
  USE_COORDINATOR=1  # make predict_release.py call into this flow
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from ..agents.base import AgentResult
from ..agents.coordinator import AgentCoordinator
from ..agents.master import run_master_flow
from ..agents.chart_vision_agent import ChartVisionAgent
from ..agents.memory_guardian_agent import MemoryGuardianAgent
from ..agents.technical_synthesis_agent import TechnicalSynthesisAgent
from ..data.ingest_news import IngestNewsInput
from ..data.ingest_news import run as run_news
from ..data.ingest_onchain import IngestOnchainInput
from ..data.ingest_onchain import run as run_onchain
from ..data.ingest_order_flow import IngestOrderFlowInput
from ..data.ingest_order_flow import run as run_orderflow
from ..data.ingest_orderbook import IngestOrderbookInput
from ..data.ingest_orderbook import run as run_orderbook
from ..data.ingest_prices import IngestPricesInput
from ..data.ingest_prices import run as run_prices
from ..data.ingest_prices_lowtf import IngestPricesLowTFInput
from ..data.ingest_prices_lowtf import run as run_prices_lowtf
from ..ensemble.ensemble_rank import EnsembleInput
from ..ensemble.ensemble_rank import run as run_ensemble
from ..features.features_calc import FeaturesCalcInput
from ..features.features_calc import run as run_features
from ..infra.db import (
    fetch_model_trust_regime,
    fetch_predictions_for_cv,
    fetch_recent_predictions,
    fetch_user_insights_recent,
    get_data_source_trust,
    insert_agent_metric,
    insert_backtest_result,
    upsert_agent_prediction,
    upsert_ensemble_weights,
    upsert_explanations,
    upsert_features_snapshot,
    upsert_prediction,
    upsert_regime,
    upsert_scenarios,
    upsert_similar_windows,
    upsert_trade_suggestion,
    upsert_validation_report,
)
from ..infra.health import start_background as start_health_server
from ..infra.logging_config import init_logging
from ..infra.metrics import push_durations, push_values, timed
from ..infra.obs import init_sentry
from ..infra.run_lock import acquire_release_lock
from ..models.models import ModelsInput
from ..models.models import run as run_models
from ..reasoning.debate_arbiter import debate, multi_debate
from ..reasoning.explain import explain_short
from ..regime.predictor_ml import predict as predict_regime_ml
from ..regime.regime_detect import detect as detect_regime
from ..reporting.charts import plot_price_with_levels
from ..scenarios.scenario_helper import run_llm_or_fallback as run_scenarios
from ..similarity.similar_past import SimilarPastInput
from ..similarity.similar_past import run as run_similar
from ..trading.trade_recommend import TradeRecommendInput
from ..trading.trade_recommend import run as run_trade
from ..trading.verifier import verify

try:
    from ..data.ingest_futures import IngestFuturesInput
    from ..data.ingest_futures import run as run_futures
except ImportError:  # pragma: no cover
    IngestFuturesInput = None  # type: ignore[assignment]
    run_futures = None  # type: ignore[assignment]
try:
    from ..data.ingest_social import IngestSocialInput
    from ..data.ingest_social import run as run_social
except ImportError:  # pragma: no cover
    IngestSocialInput = None  # type: ignore[assignment]
    run_social = None  # type: ignore[assignment]
try:
    from ..data.ingest_altdata import IngestAltDataInput
    from ..data.ingest_altdata import run as run_altdata
except ImportError:  # pragma: no cover
    IngestAltDataInput = None  # type: ignore[assignment]
    run_altdata = None  # type: ignore[assignment]

# Lightweight agent wrappers -------------------------------------------------


def _ok(name: str, output: Any, meta: dict | None = None) -> AgentResult:
    return AgentResult(name=name, ok=True, output=output, meta=meta or {})


def _fail(name: str, err: Exception) -> AgentResult:
    return AgentResult(name=name, ok=False, output=None, meta={"error": str(err)})


def _indicator_snapshot(features_path: str) -> Dict[str, float]:
    if not features_path:
        return {}
    try:
        from ..infra.s3 import download_bytes
        import pyarrow as pa
        import pyarrow.parquet as pq

        raw = download_bytes(features_path)
        table = pq.read_table(pa.BufferReader(raw), columns=[
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
        ])
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


class PricesAgent:
    """Ingest price data.

    Payload: ``IngestPricesInput``.
    Env flags: none.
    """

    name = "prices"
    priority = 10

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            out = run_prices(IngestPricesInput(**payload))
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class NewsAgent:
    """Fetch news articles.

    Payload keys: run_id, slot, time_window_hours, query.
    Env flags: none.
    """

    name = "news"
    priority = 20

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {run_id, slot, time_window_hours, query}
        try:
            out = run_news(IngestNewsInput(**payload))
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class OrderbookAgent:
    """Collect order book snapshots.

    Payload: ``IngestOrderbookInput``.
    Env flag ``ENABLE_ORDERBOOK`` controls execution.
    """

    name = "orderbook"
    priority = 25

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            if os.getenv("ENABLE_ORDERBOOK", "0") not in {"1", "true", "True"}:
                return _ok(self.name, None, meta={"skipped": True})
            out = run_orderbook(IngestOrderbookInput(**payload))
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class OrderFlowAgent:
    """Stream recent trades to S3 and compute meta.

    Payload: ``IngestOrderFlowInput``.
    Env flag ``ENABLE_ORDER_FLOW`` controls execution.
    """

    name = "order_flow"
    priority = 26

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            if os.getenv("ENABLE_ORDER_FLOW", "0") not in {"1", "true", "True"}:
                return _ok(self.name, None, meta={"skipped": True})
            out = run_orderflow(IngestOrderFlowInput(**payload))
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class OnchainAgent:
    """Process on-chain metrics.

    Payload: ``IngestOnchainInput``.
    Env flag ``ENABLE_ONCHAIN`` controls execution.
    """

    name = "onchain"
    priority = 26

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            if os.getenv("ENABLE_ONCHAIN", "0") not in {"1", "true", "True"}:
                return _ok(self.name, None, meta={"skipped": True})
            out = run_onchain(IngestOnchainInput(**payload))
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class FuturesAgent:
    """Load futures data if available.

    Payload: ``IngestFuturesInput``.
    Env flag ``ENABLE_FUTURES`` controls execution.
    """

    name = "futures"
    priority = 27

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            if run_futures is None or IngestFuturesInput is None:
                return _ok(self.name, None, meta={"skipped": True})
            if os.getenv("ENABLE_FUTURES", "0") not in {"1", "true", "True"}:
                return _ok(self.name, None, meta={"skipped": True})
            out = run_futures(IngestFuturesInput(**payload))  # type: ignore[misc]
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class SocialAgent:
    """Pull social sentiment data.

    Payload: ``IngestSocialInput``.
    Env flag ``ENABLE_SOCIAL`` controls execution.
    """

    name = "social"
    priority = 28

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            if run_social is None or IngestSocialInput is None:
                return _ok(self.name, None, meta={"skipped": True})
            if os.getenv("ENABLE_SOCIAL", "0") not in {"1", "true", "True"}:
                return _ok(self.name, None, meta={"skipped": True})
            out = run_social(IngestSocialInput(**payload))  # type: ignore[misc]
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class DeepPricesAgent:
    """Ingest low timeframe price data.

    Payload: ``IngestPricesLowTFInput``.
    Env flag ``ENABLE_DEEP_PRICE`` controls execution.
    """

    name = "prices_lowtf"
    priority = 12

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            if os.getenv("ENABLE_DEEP_PRICE", "0") not in {"1", "true", "True"}:
                return _ok(self.name, None, meta={"skipped": True})
            out = run_prices_lowtf(IngestPricesLowTFInput(**payload))
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class AltDataAgent:
    """Fetch alternative datasets.

    Payload: ``IngestAltDataInput``.
    Env flag ``ENABLE_ALT_DATA`` controls execution.
    """

    name = "alt_data"
    priority = 29

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            if run_altdata is None or IngestAltDataInput is None:
                return _ok(self.name, None, meta={"skipped": True})
            if os.getenv("ENABLE_ALT_DATA", "0") not in {"1", "true", "True"}:
                return _ok(self.name, None, meta={"skipped": True})
            out = run_altdata(IngestAltDataInput(**payload))  # type: ignore[misc]
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class FeaturesAgent:
    """Compute feature matrix.

    Payload: ``FeaturesCalcInput``.
    Env flags: none.
    """

    name = "features"
    priority = 30

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            out = run_features(FeaturesCalcInput(**payload))
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class RegimeAgent:
    """Detect market regime.

    Payload requires ``features_path_s3``.
    Env flags: none.
    """

    name = "regime"
    priority = 40

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        fpath = payload["features_path_s3"]
        try:
            rml = predict_regime_ml(fpath)
            out = {
                "label": rml.label,
                "confidence": float(max(rml.proba.values() or [0.0])),
                "features": rml.features,
            }
            return _ok(self.name, out)
        except Exception:  # noqa: BLE001
            h = detect_regime(fpath)
            out = {"label": h.label, "confidence": h.confidence, "features": h.features}
            return _ok(self.name, out, meta={"fallback": True})


class SimilarPastAgent:
    """Find similar historical windows.

    Payload: ``SimilarPastInput``.
    Env flags: none.
    """

    name = "similar_past"
    priority = 45

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            res = run_similar(SimilarPastInput(**payload))
            return _ok(self.name, [r.__dict__ for r in res])
        except Exception as e:  # noqa: BLE001
            return _ok(self.name, [], meta={"error": str(e)})


class ChartVisionFlowAgent:
    name = "chart_vision"
    priority = 67

    def __init__(self) -> None:
        self._agent = ChartVisionAgent()

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            image_urls = payload.get("image_urls") or []
            if not image_urls:
                return _ok(self.name, None, meta={"skipped": True})
            res = self._agent.run(payload)
            return _ok(self.name, res.output)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class TechnicalSynthesisFlowAgent:
    name = "technical_synthesis"
    priority = 68

    def __init__(self) -> None:
        self._agent = TechnicalSynthesisAgent()

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            if not payload.get("indicators"):
                return _ok(self.name, {}, meta={"skipped": True})
            res = self._agent.run(payload)
            return _ok(self.name, res.output)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class ModelsAgent:
    """Run ML models for a given horizon.

    Payload is passed to ``ModelsInput`` with ``horizon_minutes``.
    Env flags: none.
    """

    def __init__(self, name: str, horizon_minutes: int):
        self.name = name
        self.priority = 50 if horizon_minutes <= 240 else 51
        self.horizon_minutes = horizon_minutes

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            out = run_models(
                ModelsInput(**payload, horizon_minutes=self.horizon_minutes)
            )
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class EnsembleAgent:
    """Combine model predictions via ranking.

    Payload: ``EnsembleInput``.
    Env flags: none.
    """

    def __init__(self, name: str):
        self.name = name
        self.priority = 60

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            out = run_ensemble(EnsembleInput(**payload))
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class ScenariosAgent:
    """Generate hypothetical scenarios via LLM or fallback.

    Payload forwarded to ``run_scenarios``.
    Env flags: none.
    """

    name = "scenarios"
    priority = 65

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            sc, lvls = run_scenarios(**payload)
            return _ok(self.name, {"scenarios": sc, "levels": lvls})
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class PlotAgent:
    """Produce price chart with levels.

    Payload: parameters for ``plot_price_with_levels``.
    Env flags: none.
    """

    name = "chart"
    priority = 66

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            path = plot_price_with_levels(**payload)
            return _ok(self.name, path)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class TradeAgent:
    """Recommend trade card and verify basic constraints.

    Payload: ``TradeRecommendInput`` fields.
    Env flags: none.
    """

    name = "trade"
    priority = 70

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            safe_mode = os.getenv("SAFE_MODE", "0") in {"1", "true", "True"}
            try:
                overrides = json.loads(os.getenv("SAFE_MODE_RISK_OVERRIDES", "{}"))
            except Exception:
                overrides = {}
            base_risk = float(payload.get("risk_per_trade", 0.005))
            base_leverage = float(payload.get("leverage_cap", 25.0))
            if safe_mode:
                if "risk_per_trade" in overrides:
                    base_risk = min(base_risk, float(overrides["risk_per_trade"]))
                if "leverage_cap" in overrides:
                    base_leverage = min(base_leverage, float(overrides["leverage_cap"]))
            # Construct input strictly with expected fields
            tri = TradeRecommendInput(
                current_price=float(payload.get("current_price")),
                y_hat_4h=float(payload.get("y_hat_4h")),
                interval_low=float(payload.get("interval_low")),
                interval_high=float(payload.get("interval_high")),
                proba_up=float(payload.get("proba_up")),
                atr=float(payload.get("atr")),
                regime=payload.get("regime"),
                account_equity=float(payload.get("account_equity", 1000.0)),
                risk_per_trade=base_risk,
                leverage_cap=base_leverage,
                now_ts=(
                    int(payload.get("now_ts"))
                    if payload.get("now_ts") is not None
                    else None
                ),
                tz_name=payload.get("tz_name"),
                valid_for_minutes=int(payload.get("valid_for_minutes", 90)),
                horizon_minutes=(
                    int(payload.get("horizon_minutes"))
                    if payload.get("horizon_minutes") is not None
                    else None
                ),
            )
            card = run_trade(tri)
            interval = (
                float(payload.get("interval_low")),
                float(payload.get("interval_high")),
            )
            ok, reason = verify(
                card,
                current_price=tri.current_price,
                leverage_cap=tri.leverage_cap,
                interval=interval,
                atr=tri.atr,
            )
            return _ok(self.name, {"card": card, "verified": ok, "reason": reason})
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


# Validation agents ----------------------------------------------------------


class RiskPolicyAgent:
    """Apply heuristic risk rules to a trade card.

    Payload: card, risk metrics and liquidity info.
    Env flags: ``RISK_LEVERAGE_CAP``, ``RISK_MIN_RR``, ``RISK_VAR95_WARN``.
    """

    name = "risk_policy"
    priority = 74

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {card: dict, risk: dict|None, liquidity: dict|None, validation: dict|None}
        try:
            card = payload.get("card", {}) or {}
            risk = payload.get("risk", {}) or {}
            liq = payload.get("liquidity", {}) or {}
            val = payload.get("validation", {}) or {}
            lev = float(card.get("leverage", 0.0) or 0.0)
            rr = float(card.get("rr_expected", 0.0) or 0.0)
            var95 = float((risk or {}).get("VaR95", 0.0) or 0.0)
            liq_level = str((liq or {}).get("level", "normal"))
            vstat = str((val or {}).get("status", "OK"))
            status = "OK"
            reasons: list[str] = []
            import os as _os

            if lev > float(_os.getenv("RISK_LEVERAGE_CAP", "25")):
                status = "ERROR"
                reasons.append("leverage above cap")
            if rr < float(_os.getenv("RISK_MIN_RR", "1.2")):
                status = "WARN" if status == "OK" else status
                reasons.append("low RR")
            if var95 > float(_os.getenv("RISK_VAR95_WARN", "0.02")):
                status = "WARN" if status == "OK" else status
                reasons.append("high VaR95")
            if liq_level == "dry":
                status = "WARN" if status == "OK" else status
                reasons.append("dry liquidity")
            if vstat in {"WARN", "ERROR"}:
                status = (
                    vstat
                    if vstat == "ERROR"
                    else ("WARN" if status != "ERROR" else status)
                )
                reasons.append("validation warnings")
            return _ok(
                self.name,
                {
                    "status": status,
                    "reasons": reasons,
                    "lev": lev,
                    "rr": rr,
                    "VaR95": var95,
                    "liq": liq_level,
                },
            )
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class ValidationRouterAgent:
    """Derive validation configuration based on regime and sentiment.

    Payload: regime, sentiment_index, liquidity.
    Env flags: none.
    """

    name = "validation_router"
    priority = 71

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {regime: dict, sentiment_index: dict|None, liquidity: dict|None}
        try:
            regime = payload.get("regime", {}) or {}
            vol = float(regime.get("features", {}).get("vol_pct", 0.0) or 0.0)
            sent = float(
                (payload.get("sentiment_index") or {}).get("index", 0.0) or 0.0
            )
            liq_level = (payload.get("liquidity") or {}).get("level", "normal")
            # Base thresholds
            cfg = {
                "max_leverage": 25.0,
                "min_rr": 1.2,
                "max_interval_width_pct": 0.05,
                "allow_trade": True,
            }
            # Stricter with high volatility
            if vol > 3.0:
                cfg["max_leverage"] = 10.0
                cfg["min_rr"] = 1.5
                cfg["max_interval_width_pct"] = 0.035
            if vol > 5.0:
                cfg["max_leverage"] = 5.0
                cfg["min_rr"] = 1.8
                cfg["max_interval_width_pct"] = 0.025
            # Liquidity dry market
            if str(liq_level) == "dry":
                cfg["max_leverage"] = min(cfg["max_leverage"], 5.0)
                cfg["allow_trade"] = False
            # Strong negative sentiment in up regime => caution
            if regime.get("label") in {"trend_up", "healthy_rise"} and sent < -0.3:
                cfg["min_rr"] = max(cfg["min_rr"], 1.8)
            return _ok(self.name, cfg)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class TradeValidatorAgent:
    """Check trade card against router configuration.

    Payload: card, last_price, atr, cfg.
    Env flags: none.
    """

    name = "trade_validator"
    priority = 72

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {card: dict, last_price: float, atr: float, cfg: dict}
        try:
            card = payload.get("card", {}) or {}
            cfg = payload.get("cfg", {}) or {}
            lp = float(payload.get("last_price", 0.0) or 0.0)
            interval_width_pct = 0.0
            try:
                interval = float(card.get("take_profit", lp)) - float(
                    card.get("stop_loss", lp)
                )
                interval_width_pct = abs(interval) / max(1e-9, lp)
            except Exception:
                interval_width_pct = 0.0
            lev = float(card.get("leverage", 0.0) or 0.0)
            rr = float(card.get("rr_expected", 0.0) or 0.0)
            checks = []
            if lev > cfg.get("max_leverage", 25.0):
                checks.append(
                    {
                        "level": "WARN",
                        "msg": f"leverage {lev} > {cfg.get('max_leverage')}",
                    }
                )
            if rr < cfg.get("min_rr", 1.2):
                checks.append(
                    {"level": "WARN", "msg": f"rr {rr} < {cfg.get('min_rr')}"}
                )
            if interval_width_pct > cfg.get("max_interval_width_pct", 0.05):
                checks.append(
                    {
                        "level": "WARN",
                        "msg": f"interval width {interval_width_pct:.3f} too wide",
                    }
                )
            if not cfg.get("allow_trade", True):
                checks.append(
                    {
                        "level": "ERROR",
                        "msg": "router disallows trade (liquidity/volatility)",
                    }
                )
            status = "OK"
            if any(c["level"] == "ERROR" for c in checks):
                status = "ERROR"
            elif any(c["level"] == "WARN" for c in checks):
                status = "WARN"
            return _ok(
                self.name,
                {
                    "status": status,
                    "checks": checks,
                    "interval_width_pct": interval_width_pct,
                },
            )
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class RegimeValidatorAgent:
    """Validate model direction against regime and news signals.

    Payload: regime, y_hat, last_price, news_signals.
    Env flags: none.
    """

    name = "regime_validator"
    priority = 72

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {regime: dict, y_hat: float, last_price: float, news_signals: list[dict]}
        try:
            regime = payload.get("regime", {})
            y_hat = float(payload.get("y_hat", 0.0) or 0.0)
            lp = float(payload.get("last_price", 0.0) or 0.0)
            dir_up = y_hat >= lp
            label = str(regime.get("label", "range"))
            ok = True
            reasons = []
            if label == "trend_up" and not dir_up:
                ok = False
                reasons.append("models vs regime mismatch (down vs up)")
            if label == "trend_down" and dir_up:
                ok = False
                reasons.append("models vs regime mismatch (up vs down)")
            # Regulatory negative news increases severity
            news = payload.get("news_signals", []) or []
            neg_reg = any(
                "sec" in (str(s.get("title", "")) + str(s.get("topics", []))).lower()
                and str(s.get("sentiment", "")) == "negative"
                for s in news[:10]
            )
            level = "OK" if ok else ("WARN" if not ok and not neg_reg else "ERROR")
            return _ok(
                self.name,
                {
                    "status": level,
                    "ok": ok,
                    "reasons": reasons,
                    "neg_reg": bool(neg_reg),
                },
            )
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class LLMValidatorAgent:
    """Use external LLM service to vet trade card.

    Payload: card, scenarios, explanation.
    Env flags: ``ENABLE_LLM_VALIDATOR``, ``FLOWISE_VALIDATE_URL``, ``FLOWISE_TIMEOUT_SEC``.
    """

    name = "llm_validator"
    priority = 73

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {card: dict, scenarios: list, explanation: str}
        try:
            if os.getenv("ENABLE_LLM_VALIDATOR", "0") not in {"1", "true", "True"}:
                return _ok(self.name, None, meta={"skipped": True})
            url = os.getenv("FLOWISE_VALIDATE_URL")
            if not url:
                return _ok(self.name, None, meta={"skipped": True})
            import requests

            body = {
                "card": payload.get("card", {}),
                "scenarios": payload.get("scenarios", [])[:5],
                "explanation": payload.get("explanation", ""),
            }
            r = requests.post(
                url, json=body, timeout=float(os.getenv("FLOWISE_TIMEOUT_SEC", "15"))
            )
            if r.status_code != 200:
                return _ok(self.name, None, meta={"error": f"status {r.status_code}"})
            data = r.json() or {}
            return _ok(self.name, data)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class RiskEstimatorAgent:
    """Estimate VaR/ES and volatility from features.

    Payload requires ``features_path_s3``.
    Env flags: none.
    """

    name = "risk_estimator"
    priority = 66

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {features_path_s3: str}
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            from ..infra.s3 import download_bytes

            raw = download_bytes(payload["features_path_s3"])
            df = pq.read_table(pa.BufferReader(raw)).to_pandas().sort_values("ts")
            close = df["close"].astype(float)
            ret = close.pct_change().dropna()
            if len(ret) < 50:
                return _ok(self.name, {"status": "insufficient"})
            import numpy as np

            # VaR/ES 95%
            alpha = 0.95
            q = float(np.quantile(ret, 1 - alpha))
            var = -q
            es = float(-ret[ret <= q].mean()) if (ret <= q).any() else var
            # Volatility (annualized assuming 365*24*12 ~ minutes? Here df likely 1m; approximate daily vol)
            vol = float(ret.std(ddof=0) * np.sqrt(60 * 24))
            # Drawdown (rolling max)
            wealth = (1 + ret).cumprod()
            peak = wealth.cummax()
            dd = wealth / peak - 1.0
            max_dd = float(dd.min())
            return _ok(
                self.name,
                {"VaR95": var, "ES95": es, "vol_daily": vol, "max_drawdown": max_dd},
            )
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class ModelsCVAgent:
    """Compute CV metrics for recent model predictions.

    Payload: horizons list.
    Env flag ``CCXT_PROVIDER`` selects exchange for prices.
    """

    name = "models_cv"
    priority = 20  # run early, independent

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {horizons: list[str]}
        try:
            import ccxt  # type: ignore

            ex = getattr(ccxt, os.getenv("CCXT_PROVIDER", "binance"))(
                {"enableRateLimit": True}
            )
            horizons = payload.get("horizons", ["4h", "12h"]) or ["4h", "12h"]

            def get_y_true(ts_iso: str, horizon: str) -> Optional[float]:
                try:
                    ts = pd.Timestamp(ts_iso, tz="UTC").to_pydatetime()
                    ahead = ts + (
                        timedelta(hours=4) if horizon == "4h" else timedelta(hours=12)
                    )
                    # Fetch nearest candle 1m
                    market = "BTC/USDT"
                    ohlcv = ex.fetch_ohlcv(
                        market,
                        timeframe="1m",
                        since=int((ahead.timestamp() - 60 * 5) * 1000),
                        limit=10,
                    )
                    if not ohlcv:
                        return None
                    # choose candle closest to ahead
                    tgt = int(ahead.timestamp() * 1000)
                    row = min(ohlcv, key=lambda r: abs(r[0] - tgt))
                    return float(row[4])
                except Exception:
                    return None

            import numpy as np
            import pandas as pd

            for hz in horizons:
                preds = fetch_predictions_for_cv(
                    hz,
                    min_age_hours=1.0 if hz == "1h" else (4.0 if hz == "4h" else 12.0),
                )
                y_true = []
                y_pred = []
                for (
                    run_id,
                    horizon,
                    created_at,
                    y_hat,
                    pi_low,
                    pi_high,
                    per_model,
                ) in preds:
                    yt = get_y_true(created_at, horizon)
                    if yt is None:
                        continue
                    y_true.append(yt)
                    y_pred.append(float(y_hat or 0.0))
                if y_true and y_pred:
                    yt = np.asarray(y_true)
                    yp = np.asarray(y_pred)
                    smape = float(
                        np.mean(np.abs(yp - yt) / (np.abs(yp) + np.abs(yt) + 1e-9))
                    )
                    mae = float(np.mean(np.abs(yp - yt)))
                    # Directional accuracy: compare sign of next-step actual change vs predicted change (relative to previous actual)
                    if len(yt) > 1:
                        actual_dir = np.sign(yt[1:] - yt[:-1])
                        pred_dir = np.sign(yp[1:] - yt[:-1])
                        da = float(np.mean(actual_dir == pred_dir))
                    else:
                        da = 0.0
                    insert_agent_metric(
                        "models_cv", f"smape_{hz}", smape, labels={"horizon": hz}
                    )
                    insert_agent_metric(
                        "models_cv", f"mae_{hz}", mae, labels={"horizon": hz}
                    )
                    insert_agent_metric(
                        "models_cv", f"da_{hz}", da, labels={"horizon": hz}
                    )
            return _ok(self.name, {"status": "ok"})
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class BacktestAgent:
    """Run simple breakout backtest on features.

    Payload requires ``features_path_s3``.
    Env flags: none.
    """

    name = "backtest_validator"
    priority = 21

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {features_path_s3: str}
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            from ..infra.s3 import download_bytes

            raw = download_bytes(payload["features_path_s3"])
            df = pq.read_table(pa.BufferReader(raw)).to_pandas().sort_values("ts")
            # Simple breakout strategy on Heikin-Ashi close with ATR
            ha = df.get("ha_close") if "ha_close" in df.columns else df["close"]
            atr = df.get("atr_14").astype(float)
            close = df["close"].astype(float)
            sig_up = (ha > ha.rolling(10).mean()).astype(int)
            sig_dn = (ha < ha.rolling(10).mean()).astype(int)
            pos = 0
            eq = 1.0
            for i in range(20, len(df)):
                if pos == 0:
                    if sig_up.iloc[i] == 1:
                        pos = 1
                        entry = close.iloc[i]
                        sl = entry - 1.5 * atr.iloc[i]
                        tp = entry + 1.5 * atr.iloc[i]
                    elif sig_dn.iloc[i] == 1:
                        pos = -1
                        entry = close.iloc[i]
                        sl = entry + 1.5 * atr.iloc[i]
                        tp = entry - 1.5 * atr.iloc[i]
                else:
                    c = close.iloc[i]
                    if pos == 1 and (c <= sl or c >= tp):
                        eq *= tp / entry if c >= tp else sl / entry
                        pos = 0
                    elif pos == -1 and (c >= sl or c <= tp):
                        eq *= (entry / tp) if c <= tp else (entry / sl)
                        pos = 0
            metrics = {"equity": float(eq)}
            insert_backtest_result({"strategy": "ha_breakout"}, metrics)
            return _ok(self.name, metrics)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


# Additional quality/risk agents (heuristic P1 versions) ----------------------


class SentimentQualityAgent:
    """Assess reliability of news sentiment signals.

    Payload: list of news signals.
    Env flags: none.
    """

    name = "sentiment_quality"
    priority = 27

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {news_signals: list[dict]}
        try:
            sigs = payload.get("news_signals", []) or []
            confs = [
                float(s.get("confidence", 0.0))
                for s in sigs
                if s.get("confidence") is not None
            ]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            known_sources = {
                "coindesk",
                "reuters",
                "bloomberg",
                "the block",
                "cointelegraph",
            }
            share_known = 0.0
            if sigs:
                share_known = sum(
                    1 for s in sigs if str(s.get("source", "")).lower() in known_sources
                ) / len(sigs)
            out = {
                "avg_conf": round(avg_conf, 3),
                "n": len(sigs),
                "share_known": round(share_known, 3),
            }
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class AnomalyAgent:
    """Detect statistical anomalies in price returns.

    Payload requires ``features_path_s3``.
    Env flags: none.
    """

    name = "anomaly_detection"
    priority = 46

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {features_path_s3: str}
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            from ..infra.s3 import download_bytes

            raw = download_bytes(payload["features_path_s3"])
            df = pq.read_table(pa.BufferReader(raw)).to_pandas().sort_values("ts")
            ret = df["close"].pct_change().fillna(0.0)
            tail = ret.tail(500)
            if len(tail) < 20:
                return _ok(self.name, {"is_anomaly": False, "zscore": 0.0})
            mu, sigma = float(tail.mean()), float(tail.std(ddof=0) + 1e-9)
            z = float((tail.iloc[-1] - mu) / sigma)
            is_anom = abs(z) > 3.5
            return _ok(self.name, {"is_anomaly": bool(is_anom), "zscore": round(z, 3)})
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class AnomalyMLAgent:
    """Detect anomalies using isolation forest on features.

    Payload requires ``features_path_s3``.
    Env flags: none.
    """

    name = "anomaly_ml"
    priority = 47

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {features_path_s3: str}
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            from sklearn.ensemble import IsolationForest  # type: ignore

            from ..infra.s3 import download_bytes

            raw = download_bytes(payload["features_path_s3"])
            df = pq.read_table(pa.BufferReader(raw)).to_pandas().sort_values("ts")
            # Build feature matrix from selected columns
            cols = [
                c
                for c in df.columns
                if c not in {"ts"} and df[c].dtype.kind in ("i", "f")
            ]
            X = df[cols].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
            # Fit on last N window and score last row
            N = min(500, len(X))
            if N < 50:
                return _ok(self.name, {"is_anomaly": False, "score": 0.0})
            Xw = X.iloc[-N:]
            clf = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
            clf.fit(Xw)
            score = float(clf.decision_function([X.iloc[-1].values])[0])
            is_anom = score < -0.05
            return _ok(
                self.name, {"is_anomaly": bool(is_anom), "score": round(score, 4)}
            )
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class CrisisAgent:
    """Estimate crisis risk from regime, order book and news.

    Payload: regime, orderbook_meta and news_signals.
    Env flags: none.
    """

    name = "crisis_detection"
    priority = 67

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {regime: dict, orderbook_meta: dict|None, news_signals: list[dict]}
        try:
            regime = payload.get("regime", {})
            ob = payload.get("orderbook_meta") or {}
            news = payload.get("news_signals", [])
            vol_pct = float(regime.get("features", {}).get("vol_pct", 0.0))
            imb = float(ob.get("imbalance", 0.0) or 0.0)
            neg = sum(1 for s in news if s.get("sentiment") == "negative")
            pos = sum(1 for s in news if s.get("sentiment") == "positive")
            risk = 0
            if vol_pct > 2.5:
                risk += 1
            if abs(imb) > 0.4:
                risk += 1
            if neg > pos:
                risk += 1
            level = "low"
            if risk >= 3:
                level = "high"
            elif risk == 2:
                level = "medium"
            return _ok(
                self.name,
                {
                    "risk_level": level,
                    "vol_pct": vol_pct,
                    "ob_imbalance": imb,
                    "neg_over_pos": neg - pos,
                },
            )
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class SentimentIndexAgent:
    """Combine news and social sentiment into an index.

    Payload: news_signals and social_signals.
    Env flags: none.
    """

    name = "sentiment_index"
    priority = 29

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {news_signals: list[dict], social_signals: list[dict]}
        try:
            news = payload.get("news_signals", []) or []
            social = payload.get("social_signals", []) or []

            # News: impact-weighted sentiment (+1/-1)
            def sign(lbl: str) -> float:
                lbl = (lbl or "").lower()
                return (
                    1.0 if lbl == "positive" else (-1.0 if lbl == "negative" else 0.0)
                )

            n_val = 0.0
            n_w = 0.0
            for s in news:
                w = float(s.get("impact_score", 0.0) or 0.0) * float(
                    s.get("confidence", 0.5) or 0.5
                )
                n_val += sign(s.get("sentiment")) * w
                n_w += w
            news_idx = (n_val / n_w) if n_w > 0 else 0.0
            # Social: average sentiment score scaled by volume rate if available
            s_scores = [float(x.get("score", 0.0) or 0.0) for x in social]
            social_idx = (sum(s_scores) / max(1, len(s_scores))) if s_scores else 0.0
            index = 0.6 * news_idx + 0.4 * social_idx
            out = {
                "index": round(float(index), 4),
                "news_component": round(float(news_idx), 4),
                "social_component": round(float(social_idx), 4),
                "n_news": len(news),
                "n_social": len(social),
            }
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class LiquidityAgent:
    """Evaluate market liquidity from alt data.

    Payload requires ``alt_path_s3``.
    Env flags: ``LIQUIDITY_MIN_USD``, ``LIQUIDITY_MAX_SLIP_BPS``.
    """

    name = "liquidity_analysis"
    priority = 68

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {alt_path_s3: str}
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            from ..infra.s3 import download_bytes

            key = payload.get("alt_path_s3")
            if not key:
                return _ok(self.name, {"status": "no-data"})
            raw = download_bytes(key)
            df = pq.read_table(pa.BufferReader(raw)).to_pandas()
            liq_usd = float(
                df.loc[df["metric"] == "liquidity_total_24h_usd", "value"]
                .tail(1)
                .mean()
                or 0.0
            )
            slip_cols = [
                c
                for c in df["metric"].unique()
                if str(c).startswith("slippage_bps_avg_")
            ]
            slip_vals = []
            for c in slip_cols:
                val = df.loc[df["metric"] == c, "value"].tail(1).values
                if len(val):
                    slip_vals.append(float(val[0]))
            slip_bps = float(sum(slip_vals) / len(slip_vals)) if slip_vals else 0.0
            min_liq = float(os.getenv("LIQUIDITY_MIN_USD", "1e9"))
            max_slip = float(os.getenv("LIQUIDITY_MAX_SLIP_BPS", "15"))
            level = "normal"
            if liq_usd < min_liq or slip_bps > max_slip:
                level = "dry"
            return _ok(
                self.name,
                {"liq_24h_usd": liq_usd, "slip_bps": slip_bps, "level": level},
            )
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class ContrarianAgent:
    """Heuristic contrarian indicator from funding and sentiment.

    Payload: futures_path_s3, sentiment_index, regime.
    Env flags: none.
    """

    name = "contrarian_signal"
    priority = 69

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {futures_path_s3: str, sentiment_index: dict|None, regime: dict|None}
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            from ..infra.s3 import download_bytes

            key = payload.get("futures_path_s3")
            sent = payload.get("sentiment_index") or {}
            regime = payload.get("regime") or {}
            if not key:
                return _ok(self.name, {"status": "no-data"})
            raw = download_bytes(key)
            df = pq.read_table(pa.BufferReader(raw)).to_pandas()
            # Take BTCUSDT funding rate latest
            fr = float(
                df.loc[df["symbol"] == "BTCUSDT", "funding_rate"]
                .dropna()
                .tail(1)
                .mean()
                or 0.0
            )
            sent_idx = float(sent.get("index", 0.0) or 0.0)
            # Regime-aware heuristic: thresholds vary by regime
            label = str(regime.get("label", "range"))
            pos_thr = 0.001 if label != "trend_up" else 0.0015
            neg_thr = -0.001 if label != "trend_down" else -0.0015
            sent_up = 0.2 if label != "trend_up" else 0.3
            sent_dn = -0.2 if label != "trend_down" else -0.3
            # If funding high positive and sentiment euphoric -> risk of pullback (contra short)
            side = "NONE"
            score = 0.0
            if fr > pos_thr and sent_idx > sent_up:
                side = "CONTRA_SHORT"
                score = min(1.0, (fr / 0.005) * 0.5 + (sent_idx) * 0.5)
            elif fr < neg_thr and sent_idx < sent_dn:
                side = "CONTRA_LONG"
                score = min(1.0, (abs(fr) / 0.005) * 0.5 + (abs(sent_idx)) * 0.5)
            return _ok(
                self.name,
                {
                    "side": side,
                    "score": round(score, 3),
                    "funding": fr,
                    "sent_index": sent_idx,
                },
            )
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class MacroAgent:
    """Flag days with important macroeconomic events.

    Payload: news_signals.
    Env flags: none.
    """

    name = "macro_analyzer"
    priority = 29

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {news_signals: list[dict]}
        try:
            news = payload.get("news_signals", []) or []
            flags = 0
            for it in news:
                t = (str(it.get("title", "")) + " " + str(it.get("topics", []))).lower()
                if any(
                    k in t
                    for k in [
                        "cpi",
                        "fomc",
                        "fed",
                        "rate hike",
                        "rate cut",
                        "nonfarm",
                        "unemployment",
                        "gdp",
                        "ppi",
                    ]
                ):
                    flags += 1
            level = "none"
            if flags >= 3:
                level = "high"
            elif flags >= 1:
                level = "medium"
            return _ok(self.name, {"macro_flags": flags, "level": level})
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class LLMForecastAgent:
    """Call external LLM to forecast price movement.

    Payload: news_top, regime, last_price, atr.
    Env flags: ``ENABLE_LLM_FORECAST``, ``FLOWISE_FORECAST_URL``, ``FLOWISE_TIMEOUT_SEC``.
    """

    name = "llm_forecast"
    priority = 55

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {news_top: list[str], regime: str, last_price: float, atr: float}
        try:
            if os.getenv("ENABLE_LLM_FORECAST", "0") not in {"1", "true", "True"}:
                return _ok(self.name, None, meta={"skipped": True})
            url = os.getenv("FLOWISE_FORECAST_URL")
            if not url:
                return _ok(self.name, None, meta={"skipped": True})
            import requests

            body = {
                "news_top": payload.get("news_top", [])[:5],
                "regime": payload.get("regime"),
                "last_price": float(payload.get("last_price", 0.0) or 0.0),
                "atr": float(payload.get("atr", 0.0) or 0.0),
            }
            r = requests.post(
                url, json=body, timeout=float(os.getenv("FLOWISE_TIMEOUT_SEC", "15"))
            )
            if r.status_code != 200:
                return _ok(self.name, None, meta={"error": f"status {r.status_code}"})
            data = r.json() or {}
            p_up = float(data.get("proba_up", 0.5) or 0.5)
            delta = float(data.get("delta", 0.0) or 0.0)
            y_hat = float(body["last_price"]) + delta
            return _ok(self.name, {"p_up": p_up, "y_hat": y_hat})
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class BehaviorAgent:
    """Analyze derivatives positioning behaviour.

    Payload requires ``alt_path_s3``.
    Env flags: none.
    """

    name = "behavior_analyzer"
    priority = 68

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {alt_path_s3: str}
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            from ..infra.s3 import download_bytes

            key = payload.get("alt_path_s3")
            if not key:
                return _ok(self.name, {"status": "no-data"})
            raw = download_bytes(key)
            df = pq.read_table(pa.BufferReader(raw)).to_pandas()
            pc = df.loc[df["metric"] == "put_call_ratio", "value"].tail(1)
            oi = df.loc[df["metric"] == "open_interest", "value"].tail(1)
            put_call = float(pc.mean() or 1.0)
            open_interest = float(oi.mean() or 0.0)
            appetite = "neutral"
            if put_call > 1.1:
                appetite = "risk_off"
            elif put_call < 0.9:
                appetite = "risk_on"
            return _ok(
                self.name,
                {
                    "put_call_ratio": put_call,
                    "open_interest": open_interest,
                    "risk_appetite": appetite,
                },
            )
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class EventImpactAgent:
    """Derive market impact hypotheses from news topics.

    Payload: news_signals.
    Env flags: none.
    """

    name = "event_impact"
    priority = 69

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {news_signals: list[dict]}
        try:
            news = payload.get("news_signals", []) or []
            impact = []
            for it in news[:20]:
                t = (str(it.get("title", "")) + " " + str(it.get("topics", []))).lower()
                if any(k in t for k in ["etf", "approval"]):
                    impact.append(
                        {"event": "ETF", "direction": "bull", "confidence": 0.7}
                    )
                if any(k in t for k in ["hack", "exploit", "breach"]):
                    impact.append(
                        {"event": "HACK", "direction": "bear", "confidence": 0.6}
                    )
                if any(k in t for k in ["fork", "hardfork"]):
                    impact.append(
                        {"event": "FORK", "direction": "volatility", "confidence": 0.5}
                    )
                if any(k in t for k in ["sec", "ban", "lawsuit", "regulation"]):
                    impact.append(
                        {"event": "REG", "direction": "bear", "confidence": 0.5}
                    )
            return _ok(self.name, {"hypothesis": impact[:5]})
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class EventStudyAgent:
    """Analyze historical price reaction to similar events.

    Payload: {event_type: str, k?: int, window_hours?: int}
    Env flags: none.
    """

    name = "event_study"
    priority = 69

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        try:
            from ..agents.event_study import EventStudyInput
            from ..agents.event_study import run as run_event

            ev = EventStudyInput(
                event_type=str(payload.get("event_type")),
                k=int(payload.get("k", 10)),
                window_hours=int(payload.get("window_hours", 24)),
                symbol=str(payload.get("symbol", "BTC/USDT")),
                provider=str(
                    payload.get("provider", os.getenv("CCXT_PROVIDER", "binance"))
                ),
            )
            out = run_event(ev)
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


class ModelTrustAgent:
    """Suggest model weighting adjustments based on regime and news context.

    Payload: preds_4h, preds_12h, regime, news_ctx.
    Env flags: ``ENABLE_REGIME_WEIGHTING``, ``NEWS_SENSITIVE_MODELS``,
    ``NEWS_CTX_K``, ``REGIME_ALPHA_DECAY_DAYS``.
    """

    name = "model_trust"
    priority = 61

    def run(self, payload: dict) -> AgentResult:  # type: ignore[override]
        # payload: {preds_4h: list[dict], preds_12h: list[dict], regime: dict|None, news_ctx: float|None}
        try:
            p4 = payload.get("preds_4h", []) or []
            p12 = payload.get("preds_12h", []) or []
            regime = (payload.get("regime") or {}).get("label") or None
            news_ctx = float(payload.get("news_ctx", 0.0) or 0.0)
            use_regime = os.getenv("ENABLE_REGIME_WEIGHTING", "1") in {
                "1",
                "true",
                "True",
            }
            # Sensitive models patterns
            import fnmatch as _fnmatch

            sens_raw = os.getenv("NEWS_SENSITIVE_MODELS", "llm_news,ml_*")
            patterns = [s.strip() for s in sens_raw.split(",") if s.strip()]
            k_ctx = float(os.getenv("NEWS_CTX_K", "0.5"))

            def _alpha_map(hz: str) -> dict:
                if not (use_regime and regime):
                    return {}
                # Fetch weights with updated_at and apply recency decay
                decay_days = float(os.getenv("REGIME_ALPHA_DECAY_DAYS", "14"))
                if decay_days <= 0:
                    try:
                        return fetch_model_trust_regime(str(regime), hz) or {}
                    except Exception:
                        return {}
                try:
                    from datetime import datetime as _dt
                    from datetime import timezone as _tz

                    from ..infra.db import get_conn as _get_conn

                    now = _dt.now(_tz.utc)
                    out: dict[str, float] = {}
                    with _get_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT model, weight, updated_at FROM model_trust_regime WHERE regime_label=%s AND horizon=%s",
                                (str(regime), hz),
                            )
                            for m, w, upd in cur.fetchall() or []:
                                try:
                                    age_days = max(
                                        0.0, (now - upd).total_seconds() / 86400.0
                                    )
                                    k = math.exp(-age_days / max(1e-6, decay_days))
                                except Exception:
                                    k = 1.0
                                base = float(w or 1.0)
                                # Decay towards 1.0 as staleness grows
                                out[str(m)] = float(1.0 + (base - 1.0) * k)
                    return out
                except Exception:
                    try:
                        return fetch_model_trust_regime(str(regime), hz) or {}
                    except Exception:
                        return {}

            def _gamma(model: str) -> float:
                sens = any(_fnmatch.fnmatch(model, pat) for pat in patterns)
                # Event context boost (optional): if impactful events present
                evs = payload.get("events") or {}
                try:
                    hyp = evs.get("hypothesis") if isinstance(evs, dict) else []
                    ev_ctx = 0.0
                    for h in (hyp or [])[:5]:
                        conf = float(h.get("confidence", 0.0) or 0.0)
                        ev_ctx = max(ev_ctx, conf)
                except Exception:
                    ev_ctx = 0.0
                return 1.0 + ((k_ctx * news_ctx + 0.3 * ev_ctx) if sens else 0.0)

            def _event_type() -> str | None:
                try:
                    evs = payload.get("events") or {}
                    hyp = evs.get("hypothesis") if isinstance(evs, dict) else []
                    if hyp:
                        best = max(
                            hyp,
                            key=lambda h: float(
                                (h or {}).get("confidence", 0.0) or 0.0
                            ),
                        )
                        et = str(best.get("event", "")).upper()
                        return (
                            "ETF_APPROVAL"
                            if et == "ETF"
                            else ("SEC_ACTION" if et == "REG" else et)
                        )
                except Exception:
                    return None
                return None

            def suggest(preds, hz: str):
                # base inverse-smape
                base = {}
                for p in preds:
                    m = str(p.get("model"))
                    cv = p.get("cv_metrics") or {}
                    smape = float(cv.get("smape", 20.0) or 20.0)
                    base[m] = 1.0 / max(1e-3, smape)
                # apply alpha (regime) and gamma (news)
                alpha = _alpha_map(hz)
                # event-specific alpha (if any)
                alpha_evt = {}
                try:
                    et = _event_type()
                    if et:
                        from ..infra.db import fetch_model_trust_regime_event as _fmte

                        alpha_evt = _fmte(str(regime), hz, et) or {}
                except Exception:
                    alpha_evt = {}
                hits = sum(
                    1 for m, a in alpha.items() if abs(float(a or 1.0) - 1.0) > 1e-6
                )
                if hits:
                    try:
                        insert_agent_metric(
                            "model_trust",
                            "regime_alpha_hits",
                            float(hits),
                            labels={"horizon": hz, "regime": str(regime or "")},
                        )
                    except Exception:

                        logger.exception("Failed to insert regime_alpha_hits metric")

                        logger.exception(
                            "Failed to insert model_trust regime_alpha_hits metric"
                        )

                w = {}
                for m, b in base.items():
                    a = float(alpha.get(m, 1.0) or 1.0)
                    ae = float(alpha_evt.get(m, 1.0) or 1.0)
                    g = _gamma(m)
                    w[m] = max(0.0, b * a * ae * g)
                s = sum(w.values()) or 1.0
                out = {k: float(v / s) for k, v in w.items()}
                # also record news_ctx once per call
                try:
                    insert_agent_metric(
                        "model_trust",
                        "news_ctx",
                        float(max(0.0, min(1.0, news_ctx))),
                        labels={"horizon": hz},
                    )
                except Exception:

                    logger.exception("Failed to insert news_ctx metric")

                    logger.exception("Failed to insert model_trust news_ctx metric")

                return out

            out = {
                "w_suggest_4h": suggest(p4, "4h"),
                "w_suggest_12h": suggest(p12, "12h"),
            }
            return _ok(self.name, out)
        except Exception as e:  # noqa: BLE001
            return _fail(self.name, e)


# Flow builder ---------------------------------------------------------------


@dataclass
class ReleaseContext:
    run_id: str
    slot: str


def run_release_flow(
    slot: str = "manual", horizon_hours_4h: int = 4, horizon_hours_12h: int = 12
) -> Dict[str, AgentResult]:
    init_logging()
    init_sentry()
    # Delegate to MasterAgent when enabled
    if os.getenv("USE_MASTER_AGENT", "0") in {"1", "true", "True"}:
        logger.info("USE_MASTER_AGENT=1 → delegating to MasterAgent.run_master_flow()")
        try:
            run_master_flow(slot=slot)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"MasterAgent execution failed: {e}")
        return {}
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")

    acquired, lock_key = acquire_release_lock(slot)
    if not acquired:
        logger.info(f"Skip run_release_flow: lock exists for {lock_key}")
        return {}

    ctx = ReleaseContext(run_id=run_id, slot=slot)
    durations: Dict[str, float] = {}

    # Build coordinator and register agents
    co = AgentCoordinator(max_retries=0)
    memory_guardian = MemoryGuardianAgent()
    co.register(ModelsCVAgent(), depends_on=[])  # periodic CV metrics
    co.register(BacktestAgent(), depends_on=["features"])  # simple backtest on features
    co.register(PricesAgent(), depends_on=[])
    co.register(DeepPricesAgent(), depends_on=[])  # independent deep price snapshot
    co.register(NewsAgent(), depends_on=[])
    co.register(OrderbookAgent(), depends_on=[])
    co.register(OrderFlowAgent(), depends_on=[])  # optional short window trades
    co.register(OnchainAgent(), depends_on=[])
    co.register(FeaturesAgent(), depends_on=["prices", "news"])  # uses price+news
    co.register(RegimeAgent(), depends_on=["features"])  # uses features
    co.register(SimilarPastAgent(), depends_on=["features"])  # optional
    co.register(AnomalyMLAgent(), depends_on=["features"])  # ML anomaly
    co.register(
        SentimentQualityAgent(), depends_on=["news"]
    )  # quality of sentiment inputs
    co.register(FuturesAgent(), depends_on=[])  # independent
    co.register(SocialAgent(), depends_on=[])  # independent
    co.register(
        SentimentIndexAgent(), depends_on=["news", "social"]
    )  # aggregated sentiment
    co.register(AltDataAgent(), depends_on=[])  # independent alt data
    co.register(MacroAgent(), depends_on=["news"])  # macro flags from news
    co.register(LiquidityAgent(), depends_on=["alt_data"])  # needs alt_data parquet
    co.register(
        ContrarianAgent(), depends_on=["futures", "sentiment_index"]
    )  # funding + sentiment (regime-aware later)
    co.register(
        LLMForecastAgent(), depends_on=["models_4h", "regime"]
    )  # optional LLM forecast
    co.register(BehaviorAgent(), depends_on=["alt_data"])  # risk appetite from options
    co.register(EventImpactAgent(), depends_on=["news"])  # event impact hypotheses
    co.register(
        EventStudyAgent(), depends_on=["news"]
    )  # event study on high-impact facts
    co.register(
        ModelsAgent("models_4h", horizon_minutes=horizon_hours_4h * 60),
        depends_on=["features"],
    )  # 4h
    co.register(
        ModelsAgent("models_12h", horizon_minutes=horizon_hours_12h * 60),
        depends_on=["features"],
    )  # 12h
    co.register(
        EnsembleAgent("ensemble_4h"), depends_on=["models_4h"]
    )  # ensemble over model preds
    co.register(
        EnsembleAgent("ensemble_12h"), depends_on=["models_12h"]
    )  # ensemble 12h
    co.register(
        ModelTrustAgent(), depends_on=["models_4h", "models_12h"]
    )  # suggest weights
    co.register(
        ScenariosAgent(), depends_on=["features"]
    )  # uses features/current price+atr via payload
    co.register(
        RiskEstimatorAgent(), depends_on=["features"]
    )  # risk metrics from history
    co.register(PlotAgent(), depends_on=["scenarios"])  # needs scenarios+levels
    co.register(ChartVisionFlowAgent(), depends_on=["chart"])  # chart review
    co.register(
        TechnicalSynthesisFlowAgent(), depends_on=["features", "chart_vision"]
    )  # combined TA verdict
    co.register(
        TradeAgent(), depends_on=["ensemble_4h", "models_4h", "regime"]
    )  # trade from 4h
    co.register(
        RiskPolicyAgent(), depends_on=["trade", "risk_estimator"]
    )  # risk policy check

    # Prepare payloads per agent
    end_ts = int(now.timestamp())
    start_ts = end_ts - 72 * 3600

    payloads: Dict[str, Dict[str, Any]] = {
        "prices": {
            "run_id": ctx.run_id,
            "slot": ctx.slot,
            "symbols": ["BTCUSDT"],
            "start_ts": start_ts,
            "end_ts": end_ts,
            "provider": os.getenv("CCXT_PROVIDER", "binance"),
            "timeframe": os.getenv("CCXT_TIMEFRAME", "1m"),
        },
        "prices_lowtf": {
            "run_id": ctx.run_id,
            "slot": ctx.slot,
            "symbol": "BTCUSDT",
            "provider": os.getenv(
                "LOWTF_PROVIDER", os.getenv("CCXT_PROVIDER", "binance")
            ),
            "timeframe": os.getenv("LOWTF_TIMEFRAME", "5s"),
            "window_minutes": int(os.getenv("LOWTF_WINDOW_MIN", "30")),
        },
        "news": {
            "run_id": ctx.run_id,
            "slot": ctx.slot,
            "time_window_hours": int(os.getenv("NEWS_WINDOW_H", "12")),
            "query": os.getenv("NEWS_QUERY", "bitcoin OR BTC"),
        },
        "orderbook": {
            "run_id": ctx.run_id,
            "slot": ctx.slot,
            "symbol": "BTCUSDT",
            "provider": os.getenv("CCXT_PROVIDER", "binance"),
            "depth": int(os.getenv("ORDERBOOK_DEPTH", "50")),
        },
        "order_flow": {
            "run_id": ctx.run_id,
            "slot": ctx.slot,
            "symbol": "BTCUSDT",
            "provider": os.getenv("CCXT_PROVIDER", "binance"),
            "window_sec": int(os.getenv("ORDERFLOW_WINDOW_SEC", "60")),
        },
        "onchain": {
            "run_id": ctx.run_id,
            "slot": ctx.slot,
            "asset": "BTC",
        },
        "futures": {
            "run_id": ctx.run_id,
            "slot": ctx.slot,
            "symbols": ["BTCUSDT"],
            "provider": os.getenv(
                "FUTURES_PROVIDER", os.getenv("CCXT_PROVIDER", "binance")
            ),
        },
        "social": {
            "run_id": ctx.run_id,
            "slot": ctx.slot,
            "query": os.getenv("SOCIAL_QUERY", "bitcoin OR btc"),
            "subreddits": os.getenv(
                "SOCIAL_SUBREDDITS", "bitcoin,cryptocurrency"
            ).split(","),
            "max_items": int(os.getenv("SOCIAL_MAX_ITEMS", "100")),
        },
        "alt_data": {
            "run_id": ctx.run_id,
            "slot": ctx.slot,
            "symbol": "BTCUSDT",
            "google_queries": os.getenv(
                "ALT_GOOGLE_QUERIES", "bitcoin,btc price"
            ).split(","),
            "exchanges": os.getenv("ALT_EXCHANGES", "binance,bybit,okx,kraken").split(
                ","
            ),
            "order_size_usd": float(os.getenv("ALT_ORDER_SIZE_USD", "100000")),
        },
    }

    # Run first wave to get prices/news for features payload
    with timed(durations, "coordinator"):
        results = co.run(payloads)

    # Compose features payload from outputs (include optional OB/OnChain/Social)
    prices_out = results["prices"].output
    news_out = results["news"].output
    orderbook_out = (
        results.get("orderbook").output if results.get("orderbook") else None
    )
    orderflow_out = (
        results.get("order_flow").output if results.get("order_flow") else None
    )
    onchain_out = results.get("onchain").output if results.get("onchain") else None
    social_out = results.get("social").output if results.get("social") else None
    news_list = (
        [s.model_dump() for s in getattr(news_out, "news_signals", [])]
        if news_out
        else []
    )
    source_trust: Dict[str, float] = {}
    for sig in news_list:
        try:
            src_name = str(sig.get("source") or sig.get("provider") or "").strip()
        except Exception:
            src_name = ""
        if not src_name or src_name in source_trust:
            continue
        trust_val = get_data_source_trust(src_name)
        if trust_val is None and sig.get("provider"):
            trust_val = get_data_source_trust(str(sig.get("provider")))
        if trust_val is not None:
            source_trust[src_name] = float(trust_val)
    news_facts = [f for f in (getattr(news_out, "news_facts", []) or [])]
    onchain_list = (
        [s.model_dump() for s in getattr(onchain_out, "onchain_signals", [])]
        if onchain_out
        else []
    )
    social_list = (
        [s.model_dump() for s in getattr(social_out, "signals", [])]
        if social_out
        else []
    )
    orderbook_meta = getattr(orderbook_out, "meta", None) if orderbook_out else None
    # Extract macro flags from news (simple keyword scan)
    macro_flags: list[str] = []
    try:
        kws = {
            "CPI": ["cpi", "inflation"],
            "FOMC": ["fomc", "fed", "rate hike", "rate cut", "interest rate"],
            "Jobs": ["payroll", "nonfarm", "unemployment"],
            "GDP": ["gdp"],
            "PPI": ["ppi"],
        }
        text_list = [str(it.get("title", "")).lower() for it in news_list]
        for flag, words in kws.items():
            if any(any(w in t for w in words) for t in text_list):
                macro_flags.append(flag)
    except Exception:
        macro_flags = []

    payloads["features"] = {
        "prices_path_s3": getattr(prices_out, "prices_path_s3", None),
        "news_signals": news_list,
        "news_facts": news_facts,
        "orderbook_meta": orderbook_meta,
        "orderflow_path_s3": (
            getattr(orderflow_out, "trades_path_s3", None) if orderflow_out else None
        ),
        "onchain_signals": onchain_list,
        "social_signals": social_list,
        "macro_flags": macro_flags,
        "run_id": ctx.run_id,
        "slot": ctx.slot,
    }

    # Enrich payloads for downstream agents now that we know features
    # Re-run dependent agents only: features, regime, similar, models, ensembles, scenarios, chart, trade
    for name in [
        "features",
        "regime",
        "similar_past",
        "models_4h",
        "models_12h",
        "ensemble_4h",
        "ensemble_12h",
        "scenarios",
        "chart",
        "chart_vision",
        "technical_synthesis",
        "trade",
        "risk_policy",
        "backtest_validator",
    ]:
        durations[name] = 0.0  # initialize

    # features
    with timed(durations, "features"):
        results["features"] = co._agents["features"].run(payloads["features"])  # type: ignore[attr-defined]
    f_out = results["features"].output
    try:
        upsert_features_snapshot(
            ctx.run_id,
            getattr(f_out, "snapshot_ts", datetime.now(timezone.utc).isoformat()),
            getattr(f_out, "features_path_s3", ""),
        )
    except Exception:
        logger.exception("Failed to upsert features snapshot")

    # regime
    with timed(durations, "regime"):
        results["regime"] = co._agents["regime"].run({"features_path_s3": getattr(f_out, "features_path_s3", "")})  # type: ignore[attr-defined]
    regime = results["regime"].output or {
        "label": "range",
        "confidence": 0.5,
        "features": {},
    }
    try:
        upsert_regime(
            ctx.run_id,
            regime["label"],
            float(regime["confidence"]),
            regime.get("features", {}),
        )
    except Exception:
        logger.exception("Failed to upsert regime")

    # backtest validator (simple breakout on features)
    try:
        with timed(durations, "backtest_validator"):
            results["backtest_validator"] = co._agents["backtest_validator"].run(
                {"features_path_s3": getattr(f_out, "features_path_s3", "")}
            )  # type: ignore[attr-defined]
    except Exception:
        logger.exception("Failed to run backtest_validator agent")

    # similar past
    with timed(durations, "similar_past"):
        results["similar_past"] = co._agents["similar_past"].run({"features_path_s3": getattr(f_out, "features_path_s3", ""), "symbol": "BTCUSDT", "k": 5})  # type: ignore[attr-defined]
    neighbors = results["similar_past"].output or []
    try:
        upsert_similar_windows(ctx.run_id, neighbors)
    except Exception:
        logger.exception("Failed to upsert similar windows")

    # sentiment quality (from news)
    try:
        sq_payload = {"news_signals": news_list}
        results["sentiment_quality"] = co._agents["sentiment_quality"].run(sq_payload)  # type: ignore[attr-defined]
        upsert_agent_prediction(
            "sentiment_quality", ctx.run_id, results["sentiment_quality"].output or {}
        )
    except Exception:
        logger.exception("Failed to process sentiment_quality agent")

    # sentiment index (news + social)
    try:
        si_payload = {"news_signals": news_list, "social_signals": social_list}
        results["sentiment_index"] = co._agents["sentiment_index"].run(si_payload)  # type: ignore[attr-defined]
        upsert_agent_prediction(
            "sentiment_index", ctx.run_id, results["sentiment_index"].output or {}
        )
    except Exception:
        logger.exception("Failed to process sentiment_index agent")

    # models 4h/12h
    with timed(durations, "models_4h"):
        results["models_4h"] = co._agents["models_4h"].run({"features_path_s3": getattr(f_out, "features_path_s3", "")})  # type: ignore[attr-defined]
    with timed(durations, "models_12h"):
        results["models_12h"] = co._agents["models_12h"].run({"features_path_s3": getattr(f_out, "features_path_s3", "")})  # type: ignore[attr-defined]

    m4 = results["models_4h"].output
    m12 = results["models_12h"].output

    # Compute news context (normalized 0..1)
    def _parse_ts(s: str):
        try:
            return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc)

    try:
        import math as _m

        impact_sum = float(
            sum(float(x.get("impact_score", 0.0) or 0.0) for x in news_list)
        )
        # social burst: 30m rate vs 180m rate
        now_utc = datetime.now(timezone.utc)
        win_long = int(os.getenv("NEWS_CTX_WIN_LONG_MIN", "180"))
        win_short = int(os.getenv("NEWS_CTX_WIN_SHORT_MIN", "30"))
        long_secs = max(60.0, float(win_long) * 60.0)
        short_secs = max(60.0, float(win_short) * 60.0)
        social_recent_long = [
            s
            for s in social_list
            if (now_utc - _parse_ts(s.get("ts"))).total_seconds() <= long_secs
        ]
        social_recent_short = [
            s
            for s in social_list
            if (now_utc - _parse_ts(s.get("ts"))).total_seconds() <= short_secs
        ]
        rate_long = len(social_recent_long) / (long_secs / 60.0)
        rate_short = len(social_recent_short) / (short_secs / 60.0)
        burst = 0.0
        if rate_long > 0:
            burst = max(0.0, (rate_short / max(1e-6, rate_long)) - 1.0)
        flags_cnt = float(len(macro_flags or []))
        a = float(os.getenv("NEWS_CTX_COEFF_A", "0.1"))
        b = float(os.getenv("NEWS_CTX_COEFF_B", "0.6"))
        c = float(os.getenv("NEWS_CTX_COEFF_C", "0.5"))
        x = a * impact_sum + b * burst + c * flags_cnt
        news_ctx = 1.0 / (1.0 + _m.exp(-x))
    except Exception:
        news_ctx = 0.0

    # LLM forecast (optional) — before ensembles to include as pseudo-model
    llm_pred4 = None
    llm_pred12 = None
    try:
        news_top = []
        try:
            news_top = (
                [f"{s.title} ({s.source})" for s in news_out.news_signals[:3]]
                if news_out
                else []
            )
        except Exception:
            news_top = []
        llm_payload = {
            "news_top": news_top,
            "regime": regime.get("label"),
            "last_price": float(getattr(m4, "last_price", 0.0)),
            "atr": float(getattr(m4, "atr", 0.0)),
        }
        if os.getenv("ENABLE_LLM_FORECAST", "0") in {"1", "true", "True"}:
            results["llm_forecast"] = co._agents["llm_forecast"].run(llm_payload)  # type: ignore[attr-defined]
            upsert_agent_prediction(
                "llm_forecast", ctx.run_id, results["llm_forecast"].output or {}
            )
        lf = results.get("llm_forecast").output if results.get("llm_forecast") else None
        if isinstance(lf, dict) and lf:
            p_up = float(lf.get("p_up", 0.5) or 0.5)
            y_hat = float(
                lf.get("y_hat", getattr(m4, "last_price", 0.0))
                or getattr(m4, "last_price", 0.0)
            )
            atr = float(getattr(m4, "atr", 0.0) or 0.0)

            # intervals from ATR scaled by horizon
            def _interval(y, atr, minutes):
                import math as _mm

                band = float(atr * 1.5 * max(1.0, _mm.sqrt(minutes / 60.0)))
                return (y - band, y + band)

            l4, h4 = _interval(y_hat, atr, 240)
            l12, h12 = _interval(y_hat, atr, 720)
            cv_stub = {"source": "llm"}
            llm_pred4 = {
                "model": "llm_news",
                "y_hat": y_hat,
                "pi_low": l4,
                "pi_high": h4,
                "proba_up": p_up,
                "cv_metrics": cv_stub,
            }
            llm_pred12 = {
                "model": "llm_news",
                "y_hat": y_hat,
                "pi_low": l12,
                "pi_high": h12,
                "proba_up": p_up,
                "cv_metrics": cv_stub,
            }
    except Exception:
        llm_pred4 = None
        llm_pred12 = None

    # model trust (suggest weights)
    try:
        preds4 = [p.model_dump() for p in getattr(m4, "preds", [])]
        preds12 = [p.model_dump() for p in getattr(m12, "preds", [])]
        if llm_pred4:
            preds4.append(llm_pred4)
        if llm_pred12:
            preds12.append(llm_pred12)
        results["model_trust"] = co._agents["model_trust"].run(
            {
                "preds_4h": preds4,
                "preds_12h": preds12,
                "regime": regime,
                "news_ctx": news_ctx,
                "events": (
                    results.get("event_impact").output
                    if results.get("event_impact")
                    else None
                ),
            }
        )  # type: ignore[attr-defined]
        upsert_agent_prediction(
            "model_trust", ctx.run_id, results["model_trust"].output or {}
        )
    except Exception:
        logger.exception("Failed to process model_trust agent")

    # ensembles
    trust4 = {}
    trust12 = {}
    try:
        mt = results.get("model_trust").output or {}
        trust4 = mt.get("w_suggest_4h", {}) or {}
        trust12 = mt.get("w_suggest_12h", {}) or {}
    except Exception:
        trust4 = {}
        trust12 = {}
    with timed(durations, "ensemble_4h"):
        preds_payload_4h = [p.model_dump() for p in getattr(m4, "preds", [])]
        if llm_pred4:
            preds_payload_4h.append(llm_pred4)
        results["ensemble_4h"] = co._agents["ensemble_4h"].run({"preds": preds_payload_4h, "trust_weights": trust4, "horizon": "4h", "neighbors": neighbors})  # type: ignore[attr-defined]
    with timed(durations, "ensemble_12h"):
        preds_payload_12h = [p.model_dump() for p in getattr(m12, "preds", [])]
        if llm_pred12:
            preds_payload_12h.append(llm_pred12)
        results["ensemble_12h"] = co._agents["ensemble_12h"].run({"preds": preds_payload_12h, "trust_weights": trust12, "horizon": "12h", "neighbors": neighbors})  # type: ignore[attr-defined]
    e4 = results["ensemble_4h"].output
    e12 = results["ensemble_12h"].output

    # Run contrarian with regime context (regime-aware thresholds)
    try:
        c_payload = {
            "futures_path_s3": payloads.get("futures", {}).get("futures_path_s3"),
            "sentiment_index": (
                results.get("sentiment_index").output
                if results.get("sentiment_index")
                else None
            ),
            "regime": regime,
        }
        results["contrarian_signal"] = co._agents["contrarian_signal"].run(c_payload)  # type: ignore[attr-defined]
        upsert_agent_prediction(
            "contrarian_signal", ctx.run_id, results["contrarian_signal"].output or {}
        )
    except Exception:
        logger.exception("Failed to process contrarian_signal agent")

    # A/B challenger ensemble (optional)
    ch4 = None
    ch12 = None
    try:
        if os.getenv("ENABLE_CHALLENGER", "0") in {"1", "true", "True"}:
            mode = os.getenv("CHALLENGER_MODE", "uniform").lower()

            def _uniform(preds):
                n = max(1, len(preds))
                return {
                    "y_hat": float(sum(p.y_hat for p in preds) / n),
                    "pi_low": float(sum(p.pi_low for p in preds) / n),
                    "pi_high": float(sum(p.pi_high for p in preds) / n),
                    "proba_up": float(sum(p.proba_up for p in preds) / n),
                }

            def _stack(preds, horizon):
                try:
                    from ..ensemble.stacking import (
                        combine_with_weights,
                        suggest_weights,
                    )

                    w, n = suggest_weights(horizon)
                    if not w or n < 10:
                        return None
                    y, low, high, prob = combine_with_weights(
                        [p.model_dump() for p in preds], w
                    )
                    return {
                        "y_hat": y,
                        "pi_low": low,
                        "pi_high": high,
                        "proba_up": prob,
                        "weights": w,
                        "samples": n,
                    }
                except Exception:
                    return None

            if mode == "stacking":
                ch4 = _stack(getattr(m4, "preds", []), "4h") or _uniform(
                    getattr(m4, "preds", [])
                )
                ch12 = _stack(getattr(m12, "preds", []), "12h") or _uniform(
                    getattr(m12, "preds", [])
                )
            else:
                ch4 = _uniform(getattr(m4, "preds", []))
                ch12 = _uniform(getattr(m12, "preds", []))
    except Exception:
        ch4 = None
        ch12 = None

    # LLM forecast (optional) — already executed above for ensemble inclusion; keep here only as fallback
    try:
        if not results.get("llm_forecast"):
            news_top = []
            try:
                news_top = (
                    [f"{s.title} ({s.source})" for s in news_out.news_signals[:3]]
                    if news_out
                    else []
                )
            except Exception:
                news_top = []
            llm_payload = {
                "news_top": news_top,
                "regime": regime.get("label"),
                "last_price": float(getattr(m4, "last_price", 0.0)),
                "atr": float(getattr(m4, "atr", 0.0)),
            }
            if os.getenv("ENABLE_LLM_FORECAST", "0") in {"1", "true", "True"}:
                results["llm_forecast"] = co._agents["llm_forecast"].run(llm_payload)  # type: ignore[attr-defined]
                upsert_agent_prediction(
                    "llm_forecast", ctx.run_id, results["llm_forecast"].output or {}
                )
    except Exception:
        logger.exception("Failed to process llm_forecast agent")

    # scenarios + chart
    with timed(durations, "scenarios"):
        results["scenarios"] = co._agents["scenarios"].run(
            {
                "features_path_s3": getattr(f_out, "features_path_s3", ""),
                "current_price": float(getattr(m4, "last_price", 0.0)),
                "atr": float(getattr(m4, "atr", 0.0)),
                "slot": ctx.slot,
                "onchain_context": (onchain_list[-1] if onchain_list else {}),
                "macro_flags": macro_flags,
            }
        )  # type: ignore[attr-defined]
    sc_pack = results["scenarios"].output or {"scenarios": [], "levels": []}
    with timed(durations, "chart"):
        results["chart"] = co._agents["chart"].run(
            {
                "features_path_s3": getattr(f_out, "features_path_s3", ""),
                "title": f"BTC {ctx.slot} — 4h/12h прогноз",
                "y_hat_4h": float(getattr(e4, "y_hat", 0.0)),
                "y_hat_12h": float(getattr(e12, "y_hat", 0.0)),
                "levels": sc_pack.get("levels", []),
                "slot": ctx.slot,
            }
        )  # type: ignore[attr-defined]
    chart_s3 = results["chart"].output
    vision_output: Dict[str, Any] = {}
    try:
        with timed(durations, "chart_vision"):
            payload_cv = {
                "run_id": ctx.run_id,
                "symbol": "BTCUSDT",
                "slot": "chart_vision",
                "image_urls": [chart_s3] if chart_s3 else [],
                "regime": regime.get("label", "range"),
            }
            results["chart_vision"] = co._agents["chart_vision"].run(payload_cv)  # type: ignore[attr-defined]
            vision_output = results["chart_vision"].output or {}
    except Exception:
        vision_output = {}
    indicators_snapshot = _indicator_snapshot(getattr(f_out, "features_path_s3", ""))
    synthesis_output: Dict[str, Any] = {}
    try:
        with timed(durations, "technical_synthesis"):
            payload_ts = {
                "run_id": ctx.run_id,
                "symbol": "BTCUSDT",
                "slot": "technical_synthesis",
                "indicators": indicators_snapshot,
                "smc": (
                    results.get("smc").output
                    if results.get("smc")
                    else {}
                ),
                "vision": vision_output or {},
                "features_meta": {"features_path_s3": getattr(f_out, "features_path_s3", "")},
            }
            results["technical_synthesis"] = co._agents["technical_synthesis"].run(payload_ts)  # type: ignore[attr-defined]
            synthesis_output = results["technical_synthesis"].output or {}
    except Exception:
        synthesis_output = {}
    # Risk chart
    risk_chart_s3 = None
    try:
        from ..reporting.charts import plot_risk_breakdown

        rm = (
            results.get("risk_estimator").output
            if results.get("risk_estimator")
            else None
        )
        risk_chart_s3 = plot_risk_breakdown(
            getattr(f_out, "features_path_s3", ""),
            slot=ctx.slot,
            title=f"BTC {ctx.slot} — Risk",
            var95=(rm or {}).get("VaR95"),
            es95=(rm or {}).get("ES95"),
        )
    except Exception:
        risk_chart_s3 = None

    # anomaly detection (heuristic)
    try:
        results["anomaly_detection"] = co._agents["anomaly_detection"].run({"features_path_s3": getattr(f_out, "features_path_s3", "")})  # type: ignore[attr-defined]
        upsert_agent_prediction(
            "anomaly_detection", ctx.run_id, results["anomaly_detection"].output or {}
        )
    except Exception:
        logger.exception("Failed to process anomaly_detection agent")
    # anomaly detection (ML)
    try:
        results["anomaly_ml"] = co._agents["anomaly_ml"].run({"features_path_s3": getattr(f_out, "features_path_s3", "")})  # type: ignore[attr-defined]
        upsert_agent_prediction(
            "anomaly_ml", ctx.run_id, results["anomaly_ml"].output or {}
        )
    except Exception:
        logger.exception("Failed to process anomaly_ml agent")

    # crisis detection (heuristic)
    try:
        ob_meta = None
        try:
            ob_meta = (
                results.get("orderbook").output.meta
                if results.get("orderbook")
                and getattr(results.get("orderbook").output, "meta", None)
                else None
            )
        except Exception:
            ob_meta = None
        cd_payload = {
            "regime": regime,
            "orderbook_meta": ob_meta,
            "news_signals": news_list,
        }
        results["crisis_detection"] = co._agents["crisis_detection"].run(cd_payload)  # type: ignore[attr-defined]
        upsert_agent_prediction(
            "crisis_detection", ctx.run_id, results["crisis_detection"].output or {}
        )
    except Exception:
        logger.exception("Failed to process crisis_detection agent")

    # Liquidity analysis (needs alt_data S3)
    try:
        alt_path = None
        try:
            alt_path = (
                results.get("alt_data").output.alt_path_s3
                if results.get("alt_data")
                and getattr(results.get("alt_data").output, "alt_path_s3", None)
                else None
            )
        except Exception:
            alt_path = None
        if alt_path:
            liq_payload = {"alt_path_s3": alt_path}
            results["liquidity_analysis"] = co._agents["liquidity_analysis"].run(liq_payload)  # type: ignore[attr-defined]
            upsert_agent_prediction(
                "liquidity_analysis",
                ctx.run_id,
                results["liquidity_analysis"].output or {},
            )
    except Exception:
        logger.exception("Failed to process liquidity_analysis agent")

    # Behavior analyzer (options)
    try:
        alt_path = None
        try:
            alt_path = (
                results.get("alt_data").output.alt_path_s3
                if results.get("alt_data")
                and getattr(results.get("alt_data").output, "alt_path_s3", None)
                else None
            )
        except Exception:
            alt_path = None
        if alt_path:
            beh_payload = {"alt_path_s3": alt_path}
            results["behavior_analyzer"] = co._agents["behavior_analyzer"].run(beh_payload)  # type: ignore[attr-defined]
            upsert_agent_prediction(
                "behavior_analyzer",
                ctx.run_id,
                results["behavior_analyzer"].output or {},
            )
    except Exception:
        logger.exception("Failed to process behavior_analyzer agent")

    # Event impact (news)
    try:
        ev_payload = {"news_signals": news_list}
        results["event_impact"] = co._agents["event_impact"].run(ev_payload)  # type: ignore[attr-defined]
        upsert_agent_prediction(
            "event_impact", ctx.run_id, results["event_impact"].output or {}
        )
    except Exception:
        logger.exception("Failed to process event_impact agent")

    # Contrarian signal (funding + sentiment index)
    try:
        fut_path = None
        try:
            fut_path = (
                results.get("futures").output.futures_path_s3
                if results.get("futures")
                and getattr(results.get("futures").output, "futures_path_s3", None)
                else None
            )
        except Exception:
            fut_path = None
        sent_idx = {}
        try:
            sent_idx = results.get("sentiment_index").output or {}
        except Exception:
            sent_idx = {}
        if fut_path:
            contra_payload = {"futures_path_s3": fut_path, "sentiment_index": sent_idx}
            results["contrarian_signal"] = co._agents["contrarian_signal"].run(contra_payload)  # type: ignore[attr-defined]
            upsert_agent_prediction(
                "contrarian_signal",
                ctx.run_id,
                results["contrarian_signal"].output or {},
            )
    except Exception:
        logger.exception("Failed to process contrarian_signal agent")

    # Debate/explain (LLM optional)
    news_top = []
    try:
        news_top = (
            [f"{s.title} ({s.source})" for s in news_out.news_signals[:3]]
            if news_out
            else []
        )
    except Exception:
        logger.exception("Failed to build news_top list")
    # Build debate memory/trust (with models_cv errors)
    try:
        recent_pred = fetch_recent_predictions(limit=5, horizon="4h")
        memory = [
            f"{r[4]} run={r[0]} y_hat={r[2]:.2f} p_up={r[3]:.2f}" for r in recent_pred
        ]
    except Exception:
        memory = []
    try:
        from ..infra.db import fetch_agent_metrics as _fam

        cv = _fam("models_cv", "smape_4h%", 5)
        if cv:
            smape_vals = ", ".join(f"{ts}:{v:.3f}" for ts, v in cv)
        memory.append(f"cv smape_4h: {smape_vals}")
    except Exception:
        logger.exception("Failed to fetch agent metrics for models_cv")
    try:
        trust = (
            (results.get("model_trust").output or {}).get("w_suggest_4h", {})
            if results.get("model_trust")
            else {}
        )
    except Exception:
        trust = {}
    if vision_output:
        memory.append(
            f"chart_vision bias={vision_output.get('bias')} conf={vision_output.get('confidence')}"
        )
    if synthesis_output:
        verdict = synthesis_output.get("verdict", {})
        memory.append(
            f"tech_synthesis bias={verdict.get('bias')} conf={verdict.get('confidence')} score={verdict.get('score')}"
        )
    guardian_lessons: List[dict] = []
    try:
        planned_signal = synthesis_output.get("verdict", {}).get("bias") if synthesis_output else None
        mg_payload = {
            "scope": "trading",
            "context": {
                "planned_signal": planned_signal or ("BULLISH" if float(getattr(e4, "y_hat", 0.0)) >= 0 else "BEARISH"),
                "market_regime": regime.get("label"),
                "probability_up": float(getattr(e4, "proba_up", 0.5)),
            },
            "top_k": 3,
        }
        mg_result = memory_guardian.query(mg_payload)
        guardian_lessons = mg_result.output.get("lessons", []) if mg_result and isinstance(mg_result.output, dict) else []
    except Exception:
        guardian_lessons = []
    ta_context = {
        "technical_synthesis": synthesis_output,
        "vision": vision_output,
        "regime_features": regime.get("features", {}),
    }
    # Optional Event Study: if we have high-impact facts
    event_summary_lines: list[str] = []
    try:
        types_priority = ["HACK", "ETF_APPROVAL", "SEC_ACTION"]
        high = []
        for f in (news_facts or [])[:20]:
            t = str(f.get("type", "")).upper()
            sc = float(f.get("magnitude", 0.0) or 0.0) * float(
                f.get("confidence", 0.0) or 0.0
            )
            if t in {"HACK", "ETF_APPROVAL", "SEC_ACTION"} and sc >= 0.5:
                high.append((t, sc))
        seen = set()
        high_sorted = sorted(
            high,
            key=lambda x: (
                types_priority.index(x[0]) if x[0] in types_priority else 99,
                -x[1],
            ),
        )
        for t, _ in high_sorted[:2]:
            if t in seen:
                continue
            seen.add(t)
            res = co._agents["event_study"].run({"event_type": t, "k": 10, "window_hours": 12})  # type: ignore[attr-defined]
            out = res.output or {}
            if out and out.get("status") != "no-samples":
                avg = float(out.get("avg_change", 0.0) or 0.0)
                n = int(out.get("n", 0) or 0)
                event_summary_lines.append(
                    f"{t}: среднее изменение за 12ч ≈ {avg*100:.1f}% (n={n})"
                )
                try:
                    upsert_agent_prediction(f"event_study_{t}", ctx.run_id, out)
                except Exception:
                    pass
    except Exception:
        logger.exception("Failed to run event study agent(s)")

    # Debates (multi-agent if enabled)
    use_multi = os.getenv("ENABLE_MULTI_DEBATE", "1") in {"1", "true", "True"}
    if use_multi:
        deb_text, risk_flags = multi_debate(
            regime=regime["label"],
            news_top=news_top,
            neighbors=neighbors,
            memory=memory,
            trust=trust,
            ta=ta_context,
            lessons=guardian_lessons,
            source_trust=source_trust,
        )
    else:
        deb_text, risk_flags = debate(
            rationale_points=getattr(e4, "rationale_points", []),
            regime=regime["label"],
            news_top=news_top,
            neighbors=neighbors,
            memory=memory,
            trust=trust,
            ta=ta_context,
            lessons=guardian_lessons,
            source_trust=source_trust,
        )
    expl_text = explain_short(
        getattr(e4, "y_hat", 0.0),
        getattr(e4, "proba_up", 0.5),
        news_top,
        getattr(e4, "rationale_points", []),
    )

    # Trade card (4h primary)
    with timed(durations, "trade"):
        results["trade"] = co._agents["trade"].run(
            {
                "current_price": float(getattr(m4, "last_price", 0.0)),
                "y_hat_4h": float(getattr(e4, "y_hat", 0.0)),
                "interval_low": float(getattr(e4, "interval", [0.0, 0.0])[0]),
                "interval_high": float(getattr(e4, "interval", [0.0, 0.0])[1]),
                "proba_up": float(getattr(e4, "proba_up", 0.5)),
                "atr": float(getattr(m4, "atr", 0.0)),
                "account_equity": 1000.0,
                "risk_per_trade": 0.005,
                "leverage_cap": 25.0,
                "regime": regime["label"],
                "now_ts": int(now.timestamp()),
                "tz_name": os.environ.get("TIMEZONE", "Asia/Jerusalem"),
                "valid_for_minutes": 90,
                "horizon_minutes": horizon_hours_4h * 60,
            }
        )  # type: ignore[attr-defined]
    trade_res = results["trade"].output or {
        "card": {"side": "NO-TRADE"},
        "verified": False,
        "reason": "policy",
    }
    card = trade_res.get("card", {})
    ok = trade_res.get("verified", False)
    reason = trade_res.get("reason")

    # Validation routing + checks
    try:
        liq_out = (
            results.get("liquidity_analysis").output
            if results.get("liquidity_analysis")
            else None
        )
        si_out = (
            results.get("sentiment_index").output
            if results.get("sentiment_index")
            else None
        )
        router = co._agents["validation_router"].run({"regime": regime, "sentiment_index": si_out, "liquidity": liq_out})  # type: ignore[attr-defined]
        val_trade = co._agents["trade_validator"].run({"card": card, "last_price": float(getattr(m4, "last_price", 0.0)), "atr": float(getattr(m4, "atr", 0.0)), "cfg": router.output or {}})  # type: ignore[attr-defined]
        val_reg = co._agents["regime_validator"].run({"regime": regime, "y_hat": float(getattr(e4, "y_hat", 0.0)), "last_price": float(getattr(m4, "last_price", 0.0)), "news_signals": news_list})  # type: ignore[attr-defined]
        llm_val = co._agents["llm_validator"].run({"card": card, "scenarios": sc_pack.get("scenarios", []), "explanation": expl_text})  # type: ignore[attr-defined]
        # Aggregate status
        warns = []
        errs = []
        items = {
            "router": router.output,
            "trade": val_trade.output,
            "regime": val_reg.output,
            "llm": llm_val.output,
        }
        for name, out in [
            ("trade", val_trade.output or {}),
            ("regime", val_reg.output or {}),
            ("llm", llm_val.output or {}),
        ]:
            if isinstance(out, dict):
                if out.get("status") == "ERROR":
                    errs.append(f"{name}: error")
                elif out.get("status") == "WARN":
                    warns.append(f"{name}: warn")
        final_status = "OK" if not warns and not errs else ("ERROR" if errs else "WARN")
        upsert_validation_report(ctx.run_id, final_status, items, warns, errs)
        if final_status != "OK":
            # override verification
            ok = False
        validation_status = final_status
        validation_warns = len(warns)
        validation_errs = len(errs)
    except Exception:
        validation_status = None
        validation_warns = None
        validation_errs = None

    # Risk policy agent
    try:
        rp_payload = {
            "card": card,
            "risk": (
                results.get("risk_estimator").output
                if results.get("risk_estimator")
                else {}
            )
            or {},
            "liquidity": (
                results.get("liquidity_analysis").output
                if results.get("liquidity_analysis")
                else {}
            )
            or {},
            "validation": {"status": validation_status} if validation_status else {},
        }
        results["risk_policy"] = co._agents["risk_policy"].run(rp_payload)  # type: ignore[attr-defined]
        upsert_agent_prediction(
            "risk_policy", ctx.run_id, results["risk_policy"].output or {}
        )
    except Exception:
        logger.exception("Failed to process risk_policy agent")

    # Prepare concise message for Telegram
    try:
        from datetime import datetime, timezone

        from ..models.calibration_runtime import calibrate_proba as _calib
        from ..telegram_bot.publisher import publish_message, publish_photo_from_s3
        from ..utils.calibration import calibrate_proba_by_uncertainty

        try:
            from zoneinfo import ZoneInfo  # py>=3.9
        except Exception:
            ZoneInfo = None  # type: ignore

        # Calibrated probabilities
        e4_raw = float(getattr(e4, "proba_up", 0.5))
        e12_raw = float(getattr(e12, "proba_up", 0.5))
        e4_iso = _calib(e4_raw, "4h")
        e12_iso = _calib(e12_raw, "12h")
        e4_proba_cal = (
            calibrate_proba_by_uncertainty(
                e4_raw,
                getattr(e4, "interval", [0.0, 0.0]),
                getattr(m4, "last_price", 0.0),
                getattr(m4, "atr", 0.0),
            )
            if e4_iso == e4_raw
            else e4_iso
        )
        e12_proba_cal = (
            calibrate_proba_by_uncertainty(
                e12_raw,
                getattr(e12, "interval", [0.0, 0.0]),
                getattr(m4, "last_price", 0.0),
                getattr(m4, "atr", 0.0),
            )
            if e12_iso == e12_raw
            else e12_iso
        )

        # Sentiment / risk quick factors
        si = results.get("sentiment_index").output or {}
        riskm = results.get("risk_estimator").output or {}
        var95 = float(riskm.get("VaR95", 0.0) or 0.0)
        risk_level = "low" if var95 < 0.01 else ("medium" if var95 < 0.02 else "high")

        # Next update (local TZ)
        def _next_update_label(
            tz_name: str = os.getenv("TIMEZONE", "Europe/Moscow")
        ) -> str:
            try:
                tz = ZoneInfo(tz_name) if ZoneInfo else timezone.utc
            except Exception:
                tz = timezone.utc
            now = datetime.now(timezone.utc).astimezone(tz)
            h = now.hour
            target_h = 12 if h < 12 else 24
            next_dt = now.replace(
                hour=0 if target_h == 24 else 12, minute=0, second=0, microsecond=0
            )
            if next_dt <= now:
                # move to next slot
                add_h = 12 if h < 12 else (24 - h)
                next_dt = (now + timedelta(hours=add_h)).replace(
                    minute=0, second=0, microsecond=0
                )
            return next_dt.strftime("%d.%m %H:%M %Z")

        # Scenario brief
        sc_list = (
            sc_pack.get("scenarios", []) if isinstance(sc_pack, dict) else []
        ) or []
        sc_line = "n/a"
        if sc_list:
            sc = sc_list[0]
            try:
                sc_line = f"{sc.get('if_level', '')} → {sc.get('then_path', '')} (p={float(sc.get('prob', 0.0)):.2f})"
            except Exception:
                sc_line = "n/a"

        # Price ranges
        i4 = getattr(e4, "interval", [0.0, 0.0])
        i12 = getattr(e12, "interval", [0.0, 0.0])

        msg = []
        msg.append(f"<b>BTC Forecast — {ctx.slot}</b>")
        msg.append(f"Run: <code>{ctx.run_id}</code>")
        msg.append(
            f"Price ranges: 4h=({i4[0]:.2f}..{i4[1]:.2f}), 12h=({i12[0]:.2f}..{i12[1]:.2f})"
        )
        msg.append(f"Scenario: {sc_line}")
        msg.append(f"P(up): 4h={e4_proba_cal:.2f}, 12h={e12_proba_cal:.2f}")
        msg.append(
            f"Factors: news_idx={float(si.get('index', 0.0)):.2f}; risk={risk_level} (VaR95≈{var95:.3f})"
        )
        msg.append(f"Next update: {_next_update_label()}")

        # publish
        try:
            publish_photo_from_s3(
                chart_s3, caption="btc_forecast_caption", slot=ctx.slot
            )
            if risk_chart_s3:
                publish_photo_from_s3(
                    risk_chart_s3, caption="btc_risk_caption", slot=ctx.slot
                )
            publish_message("\n".join(msg))
        except Exception:
            logger.exception("Failed to publish forecast message")
    except Exception:
        logger.exception("Failed to prepare forecast message")

    # Persist predictions/scenarios/explanations best-effort
    try:
        per_model_4h = {
            p.model: {
                "y_hat": p.y_hat,
                "pi_low": p.pi_low,
                "pi_high": p.pi_high,
                "proba_up": p.proba_up,
                "cv": p.cv_metrics,
            }
            for p in getattr(m4, "preds", [])
        }
        per_model_12h = {
            p.model: {
                "y_hat": p.y_hat,
                "pi_low": p.pi_low,
                "pi_high": p.pi_high,
                "proba_up": p.proba_up,
                "cv": p.cv_metrics,
            }
            for p in getattr(m12, "preds", [])
        }
        upsert_prediction(
            ctx.run_id,
            "4h",
            getattr(e4, "y_hat", 0.0),
            getattr(e4, "interval", [0.0, 0.0])[0],
            getattr(e4, "interval", [0.0, 0.0])[1],
            getattr(e4, "proba_up", 0.5),
            per_model_4h,
        )
        upsert_prediction(
            ctx.run_id,
            "12h",
            getattr(e12, "y_hat", 0.0),
            getattr(e12, "interval", [0.0, 0.0])[0],
            getattr(e12, "interval", [0.0, 0.0])[1],
            getattr(e12, "proba_up", 0.5),
            per_model_12h,
        )
        upsert_ensemble_weights(
            ctx.run_id,
            getattr(e4, "weights", {})
            | {f"12h:{k}": v for k, v in getattr(e12, "weights", {}).items()},
        )
        upsert_scenarios(ctx.run_id, sc_pack.get("scenarios", []), chart_s3)
        # A/B challenger save as agent results
        try:
            if ch4:
                upsert_agent_prediction("ensemble_4h_ch", ctx.run_id, ch4)
            if ch12:
                upsert_agent_prediction("ensemble_12h_ch", ctx.run_id, ch12)
        except Exception:

            logger.exception("Failed to upsert challenger ensemble results")

            logger.exception("Failed to save challenger ensemble predictions")

        # Build improved explanation (Pro)
        try:
            si = results.get("sentiment_index").output or {}
            liq = results.get("liquidity_analysis").output or {}
            be = results.get("behavior_analyzer").output or {}
            riskm = results.get("risk_estimator").output or {}
        except Exception:
            si = {}
            liq = {}
            be = {}
            riskm = {}
        md = (
            "<b>Почему так</b>\n"
            + expl_text
            + "\n\n<b>Арбитраж</b>\n"
            + deb_text
            + "\n\n<b>Контекст</b>\n"
            f"Сентимент: idx={si.get('index',0):.2f} (news={si.get('news_component',0):.2f}, social={si.get('social_component',0):.2f})\n"
            f"Ликвидность: {liq.get('level','n/a')} (24h≈{liq.get('liq_24h_usd',0):.0f}$, slip≈{liq.get('slip_bps',0):.1f}bps)\n"
            f"Опционы: PC={be.get('put_call_ratio','n/a')} OI≈{be.get('open_interest','n/a')} риск={be.get('risk_appetite','n/a')}\n"
            f"Риск-метрики: VaR95≈{riskm.get('VaR95','n/a')}, ES95≈{riskm.get('ES95','n/a')}, maxDD≈{riskm.get('max_drawdown','n/a')}"
            + (f"\nRisk chart: {risk_chart_s3}" if risk_chart_s3 else "")
            + (
                "\n\n<b>Event Study</b>\n" + "\n".join(event_summary_lines)
                if event_summary_lines
                else ""
            )
        )
        upsert_explanations(ctx.run_id, md, risk_flags)
        upsert_trade_suggestion(ctx.run_id, card)
    except Exception:

        logger.exception("Failed to persist explanation or trade suggestion")

        logger.exception("Failed to persist explanations or trade suggestion")

    # metrics
    push_durations(job="release_flow", durations=durations, labels={"slot": ctx.slot})
    try:
        business = {
            "p_up_4h": float(getattr(e4, "proba_up", 0.5)),
            "p_up_12h": float(getattr(e12, "proba_up", 0.5)),
            "p_up_4h_cal": float(e4_proba_cal),
            "p_up_12h_cal": float(e12_proba_cal),
            "interval_width_pct_4h": float(
                (
                    getattr(e4, "interval", [0.0, 0.0])[1]
                    - getattr(e4, "interval", [0.0, 0.0])[0]
                )
                / max(1e-6, float(getattr(m4, "last_price", 0.0)))
            ),
            "atr_pct": float(
                float(getattr(m4, "atr", 0.0))
                / max(1e-6, float(getattr(m4, "last_price", 0.0)))
            ),
            "no_trade": 1.0 if (card.get("side") == "NO-TRADE" or not ok) else 0.0,
            "rr_expected": float(card.get("rr_expected") or 0.0),
            "leverage": float(card.get("leverage") or 0.0),
            "neighbors_k": float(len(neighbors)),
            "news_ctx": float(
                max(0.0, min(1.0, news_ctx)) if "news_ctx" in locals() else 0.0
            ),
        }
        # Risk metrics from RiskEstimatorAgent
        try:
            rm = (
                results.get("risk_estimator").output
                if results.get("risk_estimator")
                else None
            )
            if isinstance(rm, dict) and rm:
                for k in ["VaR95", "ES95", "vol_daily", "max_drawdown"]:
                    if k in rm and rm[k] is not None:
                        business[f"risk_{k}"] = float(rm[k])
        except Exception:

            logger.exception("Failed to collect risk estimator metrics")

            logger.exception("Failed to add risk metrics to business stats")

        # Validation summary
        if validation_status:
            business["validation_warns"] = float(validation_warns or 0)
            business["validation_errs"] = float(validation_errs or 0)
            business["validation_ok"] = 1.0 if validation_status == "OK" else 0.0
        # A/B divergence
        try:
            if ch4:
                business["ab_div_4h_abs"] = abs(
                    float(ch4.get("y_hat", 0.0)) - float(getattr(e4, "y_hat", 0.0))
                )
            if ch12:
                business["ab_div_12h_abs"] = abs(
                    float(ch12.get("y_hat", 0.0)) - float(getattr(e12, "y_hat", 0.0))
                )
        except Exception:
            logger.exception("Failed to compute A/B divergence metrics")
        # Risk policy status
        try:
            rp = (
                results.get("risk_policy").output
                if results.get("risk_policy")
                else None
            )
            if isinstance(rp, dict) and rp:
                business["risk_policy_ok"] = (
                    1.0 if str(rp.get("status")) == "OK" else 0.0
                )
        except Exception:

            logger.exception("Failed to record risk policy status")

            logger.exception("Failed to add risk_policy metrics")

        push_values(job="release_flow", values=business, labels={"slot": ctx.slot})
        # Per-model weights metrics
        try:
            w4 = getattr(e4, "weights", {}) or {}
            if w4:
                push_values(
                    job="ensemble",
                    values={f"ensemble_weight_{m}": float(v) for m, v in w4.items()},
                    labels={"horizon": "4h"},
                )
        except Exception:
            logger.exception("Failed to push 4h ensemble weights")

            logger.exception("Failed to push ensemble weights for 4h")

        try:
            w12 = getattr(e12, "weights", {}) or {}
            if w12:
                push_values(
                    job="ensemble",
                    values={f"ensemble_weight_{m}": float(v) for m, v in w12.items()},
                    labels={"horizon": "12h"},
                )
        except Exception:

            logger.exception("Failed to push 12h ensemble weights")

            logger.exception("Failed to push ensemble weights for 12h")

    except Exception:
        logger.exception("Failed to push business metrics")

    # Persist compact run summary for agent memory (best-effort)
    try:
        from ..infra.db import insert_run_summary as _ins_rs

        final_summary = {
            "slot": ctx.slot,
            "regime": (
                str(regime.get("label"))
                if isinstance(regime, dict)
                else str(getattr(regime, "label", ""))
            ),
            "regime_conf": float(
                (
                    regime.get("confidence")
                    if isinstance(regime, dict)
                    else getattr(regime, "confidence", 0.0)
                )
                or 0.0
            ),
            "e4": {
                "y_hat": float(getattr(e4, "y_hat", 0.0)),
                "proba_up": float(getattr(e4, "proba_up", 0.0)),
                "interval": list(getattr(e4, "interval", [0.0, 0.0])),
            },
            "e12": {
                "y_hat": float(getattr(e12, "y_hat", 0.0)),
                "proba_up": float(getattr(e12, "proba_up", 0.0)),
                "interval": list(getattr(e12, "interval", [0.0, 0.0])),
            },
            "risk_flags": risk_flags if isinstance(risk_flags, list) else [],
        }
        _ins_rs(ctx.run_id, final_summary, None)
    except Exception:
        logger.exception("Failed to insert run summary")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", default=os.environ.get("SLOT", "manual"))
    args = parser.parse_args()
    start_health_server()
    try:
        run_release_flow(slot=args.slot)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"agent_flow failed: {e}")


if __name__ == "__main__":
    main()
    # Merge validated user insights into news_list as additional signals
    try:
        ui = fetch_user_insights_recent(
            hours=int(os.getenv("INSIGHTS_WINDOW_H", "24")),
            min_truth=float(os.getenv("INSIGHTS_MIN_TRUTH", "0.6")),
            min_freshness=float(os.getenv("INSIGHTS_MIN_FRESH", "0.5")),
            limit=int(os.getenv("INSIGHTS_MAX", "30")),
        )
        for it in ui:
            # Heuristic sentiment
            title = (it.get("text") or "").strip()
            tl = title.lower()
            if any(w in tl for w in ["bull", "pump", "up", "рост", "памп"]):
                sentiment = "positive"
            elif any(
                w in tl
                for w in ["bear", "dump", "down", "пад", "крах", "hack", "exploit"]
            ):
                sentiment = "negative"
            else:
                sentiment = "neutral"
            w = min(
                float(it.get("score_truth", 0.0)), float(it.get("score_freshness", 0.0))
            )
            impact = 0.3 + 0.7 * max(0.0, min(1.0, w))
            ts = int((it.get("created_at") or datetime.now(timezone.utc)).timestamp())
            news_list.append(
                {
                    "ts": ts,
                    "title": title[:240],
                    "url": it.get("url") or "",
                    "source": "user_insight",
                    "sentiment": sentiment,
                    "impact_score": impact,
                }
            )
    except Exception:
        logger.warning("Failed to merge user insights; continuing")
