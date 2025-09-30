from __future__ import annotations

"""Investment arbiter — dynamic risk mediator to prevent analytical paralysis.

The arbiter analyses recent structured lessons, current regime, and base
probabilities to adjust the final conviction of trading recommendations.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..reasoning.llm import call_openai_json
from ..infra import metrics
from ..infra.db import fetch_agent_performance
from .context_builder_agent import ContextBuilderAgent
from .self_critique_agent import SelfCritiqueAgent
from .strategic.trust import TrustMonitor
from loguru import logger


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, value)))


def _normalize_agent_name(agent: str) -> str:
    agent_lc = agent.lower()
    if "ensemble" in agent_lc or "model" in agent_lc or "ml" in agent_lc:
        return "models"
    if "vision" in agent_lc or "chart" in agent_lc:
        return "vision"
    if "smc" in agent_lc:
        return "smc"
    if "debate" in agent_lc or "arbiter" in agent_lc:
        return "debate"
    if "news" in agent_lc or "sentiment" in agent_lc:
        return "news"
    return agent_lc


@dataclass
class ArbiterDecision:
    """Result of the investment arbiter evaluation."""

    proba_up: float
    success_probability: float
    risk_stance: str
    confidence_floor: Optional[float]
    notes: List[str] = field(default_factory=list)
    model_overrides: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalystReport:
    scenario: str
    probability_pct: float
    macro_summary: str
    technical_summary: str
    contradictions: List[str]
    explanation: str
    raw: Dict[str, object]
    warnings: List[str] = field(default_factory=list)
    warnings: List[str]

    def probability(self) -> float:
        return _clamp(self.probability_pct / 100.0)


@dataclass
class CritiqueReport:
    counterarguments: List[str]
    invalidators: List[str]
    missing_factors: List[str]
    probability_adjustment: float
    recommendation: str
    raw: Dict[str, object]
    warnings: List[str] = field(default_factory=list)
    warnings: List[str]


@dataclass
class ArbiterFullDecision:
    mode: str
    analysis: AnalystReport | None
    critique: CritiqueReport | None
    evaluation: ArbiterDecision
    context: Dict[str, object]


class InvestmentArbiter:
    """Meta controller for risk acceptance and dynamic weighting."""

    def __init__(
        self,
        acceptance_threshold: float = 0.7,
        high_bar: float = 0.92,
        max_penalty: float = 0.18,
        lesson_weight: float = 0.08,
        trust_weight: float = 0.12,
        trust_monitor: TrustMonitor | None = None,
        context_agent: ContextBuilderAgent | None = None,
        critique_agent: SelfCritiqueAgent | None = None,
        analyst_llm=call_openai_json,
    ) -> None:
        self.acceptance_threshold = acceptance_threshold
        self.high_bar = high_bar
        self.max_penalty = max_penalty
        self.lesson_weight = lesson_weight
        self.trust_weight = trust_weight
        self._trust_monitor = trust_monitor or TrustMonitor()
        self._context_agent = context_agent or ContextBuilderAgent()
        self._critique_agent = critique_agent or SelfCritiqueAgent()
        self._analyst_llm = analyst_llm

    # ------------------------------------------------------------------
    def decide(self, payload: dict) -> ArbiterFullDecision:
        run_id = str(payload.get("run_id") or "manual")
        planned_side = str(payload.get("planned_side") or "LONG").upper()
        regime_label = str(payload.get("regime_label") or payload.get("regime", {}).get("label") or "range")
        lessons = payload.get("lessons") or []
        model_trust = payload.get("model_trust") or {}
        risk_flags = payload.get("risk_flags") or []
        safe_mode = bool(payload.get("safe_mode", False))
        base_proba_up = float(payload.get("base_proba_up", 0.5) or 0.5)
        context_payload = payload.get("context_payload") or {}
        features_path_s3 = payload.get("features_path_s3")
        if features_path_s3 and "features_path_s3" not in context_payload:
            context_payload["features_path_s3"] = features_path_s3

        mode = self._mode_for_run(run_id)
        if mode == "legacy":
            evaluation = self.evaluate(
                base_proba_up=base_proba_up,
                side=planned_side,
                regime=regime_label,
                lessons=lessons,
                model_trust=model_trust,
                risk_flags=risk_flags,
                safe_mode=safe_mode,
            )
            return ArbiterFullDecision(
                mode=mode,
                analysis=None,
                critique=None,
                evaluation=evaluation,
                context={},
            )

        context_payload.update(
            {
                "run_id": run_id,
                "symbol": payload.get("symbol", "BTC/USDT"),
                "slot": payload.get("slot", "arbiter"),
            }
        )
        context_result = self._context_agent.run(context_payload)
        context_bundle = context_result.output.get("context") if isinstance(context_result.output, dict) else {}
        active_agents = self._extract_active_agents(context_bundle)
        analysis = self._run_analyst(planned_side, base_proba_up, regime_label, context_bundle, payload)
        critique = None
        adjusted_prob = base_proba_up
        if analysis:
            if analysis.scenario == "LONG":
                adjusted_prob = analysis.probability()
            elif analysis.scenario == "SHORT":
                adjusted_prob = 1.0 - analysis.probability()
            else:
                adjusted_prob = 0.5
            if analysis.warnings:
                warn_factor = float(os.getenv("ARB_WARNING_CONF_FACTOR", "0.9"))
                adjusted_prob = _clamp(adjusted_prob * warn_factor)
        if analysis and self._should_run_selfcritique():
            critique = self._run_critique(analysis, context_bundle)
            if critique:
                adjusted_prob = _clamp(adjusted_prob + critique.probability_adjustment / 100.0)
                if critique.warnings:
                    warn_factor = float(os.getenv("ARB_WARNING_CONF_FACTOR", "0.9"))
                    adjusted_prob = _clamp(adjusted_prob * warn_factor)

        dynamic_agent_weights = self._dynamic_agent_weights(regime_label, active_agents)

        evaluation = self.evaluate(
            base_proba_up=adjusted_prob,
            side=planned_side,
            regime=regime_label,
            lessons=lessons,
            model_trust=self._merge_agent_weights(model_trust, dynamic_agent_weights),
            risk_flags=risk_flags,
            safe_mode=safe_mode,
        )
        if analysis:
            evaluation.notes.append(f"analysis_scenario:{analysis.scenario}")
            if analysis.warnings:
                evaluation.notes.extend([f"analysis_warn:{w}" for w in analysis.warnings])
        if critique:
            evaluation.notes.append(f"critique:{critique.recommendation}")
            if critique.warnings:
                evaluation.notes.extend([f"critique_warn:{w}" for w in critique.warnings])

        context_hash = None
        tokens_estimate = None
        if context_bundle:
            try:
                tokens_estimate = float(context_bundle.get("tokens_estimate"))
            except Exception:
                tokens_estimate = None
            try:
                context_hash = hashlib.sha256(
                    json.dumps(context_bundle, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
            except Exception:
                context_hash = None

        s3_path = None
        if context_bundle and os.getenv("ARB_STORE_S3", "1") in {"1", "true", "True"}:
            try:
                from ..infra.s3 import upload_bytes

                payload_s3 = {
                    "run_id": run_id,
                    "mode": mode,
                    "analysis": analysis.raw if analysis else None,
                    "critique": critique.raw if critique else None,
                    "evaluation": {
                        "success_probability": evaluation.success_probability,
                        "risk_stance": evaluation.risk_stance,
                        "notes": evaluation.notes,
                    },
                    "context": context_bundle,
                }
                date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                slot = payload.get("slot", "arbiter")
                path = f"runs/{date_key}/{slot}/arbiter/{run_id}.json"
                upload_bytes(
                    path,
                    json.dumps(payload_s3, ensure_ascii=False).encode("utf-8"),
                    content_type="application/json",
                )
                s3_path = path
            except Exception:
                s3_path = None

        if os.getenv("ARB_LOG_TO_DB", "1") in {"1", "true", "True"}:
            try:
                from ..infra.db import upsert_arbiter_reasoning, upsert_arbiter_selfcritique

                upsert_arbiter_reasoning(
                    run_id=run_id,
                    mode=mode,
                    analysis=(analysis.raw if analysis else {}),
                    context=context_bundle or {},
                    tokens_estimate=tokens_estimate,
                    context_ref=context_hash,
                    s3_path=s3_path,
                )
                if critique:
                    upsert_arbiter_selfcritique(
                        run_id=run_id,
                        recommendation=critique.recommendation,
                        probability_delta=critique.probability_adjustment,
                        payload=critique.raw,
                    )
            except Exception:
                pass
        return ArbiterFullDecision(
            mode=mode,
            analysis=analysis,
            critique=critique,
            evaluation=evaluation,
            context=context_bundle,
        )

    def _mode_for_run(self, run_id: str) -> str:
        if os.getenv("ARB_ANALYST_ENABLED", "1") not in {"1", "true", "True"}:
            return "legacy"
        percent_raw = os.getenv("ARB_ANALYST_AB_PERCENT", "100")
        try:
            percent = max(0, min(100, int(percent_raw)))
        except Exception:
            percent = 100
        if percent >= 100:
            return "modern"
        digest = int(hashlib.sha256(run_id.encode("utf-8")).hexdigest(), 16)
        bucket = digest % 100
        return "modern" if bucket < percent else "legacy"

    def _should_run_selfcritique(self) -> bool:
        return os.getenv("ENABLE_SELF_CRITIQUE", "1") in {"1", "true", "True"}

    def _run_analyst(
        self,
        planned_side: str,
        base_proba_up: float,
        regime_label: str,
        context_bundle: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> AnalystReport | None:
        try:
            context_text = context_bundle.get("text") or ""
            summary_section = context_bundle.get("summary") or {}
            system_prompt = self._system_prompt()
            user_prompt = self._user_prompt(
                context_text,
                summary_section,
                regime_label,
                planned_side,
                base_proba_up,
            )
            model = os.getenv("OPENAI_MODEL_ANALYST", os.getenv("OPENAI_MODEL_MASTER"))
            raw = self._analyst_llm(system_prompt, user_prompt, model=model, temperature=0.2)
            if not raw or raw.get("status") == "error":
                return None
            from ..reasoning.schemas import ArbiterAnalystResponse
            from ..reasoning.validators import validate_analyst

            parsed = ArbiterAnalystResponse(**raw).sanitized()
            warnings = validate_analyst(parsed)
            strict = os.getenv("ARB_STRICT_VALIDATION", "0") in {"1", "true", "True"}
            if warnings:
                logger.warning("Analyst validation warnings: %s", warnings)
                if strict:
                    logger.warning("Strict validation triggered. Falling back to legacy.")
                    return None
            report = self._parse_analyst_response(parsed.model_dump(), planned_side, base_proba_up)
            if warnings:
                report.warnings = list(warnings)
            try:
                metrics.push_values(
                    job="investment-analyst",
                    values={
                        "probability_pct": float(report.probability_pct),
                    },
                    labels={"scenario": report.scenario, "regime": regime_label},
                )
            except Exception:
                pass
            return report
        except Exception:
            return None

    def _run_critique(self, analysis: AnalystReport, context_bundle: Dict[str, Any]) -> CritiqueReport | None:
        try:
            res = self._critique_agent.run({"analysis": analysis.raw, "context": context_bundle})
            out = res.output if isinstance(res.output, dict) else {}
            from ..reasoning.schemas import CritiqueResponse
            from ..reasoning.validators import validate_critique

            parsed = CritiqueResponse(**out).sanitized()
            warnings_list = validate_critique(parsed)
            strict = os.getenv("ARB_STRICT_VALIDATION", "0") in {"1", "true", "True"}
            if warnings_list:
                logger.warning("Critique validation warnings: %s", warnings_list)
                if strict:
                    logger.warning("Strict critique validation triggered. Ignoring critique")
                    return None
            out = parsed.model_dump()
            return CritiqueReport(
                counterarguments=[str(x) for x in out.get("counterarguments", [])],
                invalidators=[str(x) for x in out.get("invalidators", [])],
                missing_factors=[str(x) for x in out.get("missing_factors", [])],
                probability_adjustment=float(out.get("probability_adjustment", 0.0)),
                recommendation=str(out.get("recommendation", "REVISE")),
                raw=out,
                warnings=list(warnings_list),
            )
        except Exception:
            return None

    def _system_prompt(self) -> str:
        return (
            "Ты — главный инвестиционный аналитик в хедж-фонде. Проанализируй предоставленный контекст"
            " последовательно и верни JSON с полями: scenario (LONG/SHORT/FLAT), probability_pct (float),"
            " macro_summary, technical_summary, contradictions (list[str]), explanation (string)."
            " Соблюдай шаги: анализ контекста, технический синтез, поиск противоречий, формулировка гипотезы,"
            " оценка вероятности. Если есть серьёзные противоречия, вероятность не выше 80." )

    def _user_prompt(
        self,
        context_text: str,
        summary_section: Dict[str, Any],
        regime_label: str,
        planned_side: str,
        base_proba_up: float,
    ) -> str:
        return (
            f"Текущий режим: {regime_label}. Планируемая сторона: {planned_side}."
            f" Базовая вероятность моделей: {base_proba_up*100:.1f}%.\n"
            "Полный контекст ниже: \n"
            f"{context_text}"
        )

    def _parse_analyst_response(
        self,
        raw: Dict[str, Any],
        planned_side: str,
        base_proba_up: float,
    ) -> AnalystReport:
        if not raw or raw.get("status") == "error":
            default_prob = base_proba_up * 100 if planned_side == "LONG" else (1.0 - base_proba_up) * 100
            return AnalystReport(
                scenario=planned_side,
                probability_pct=float(round(default_prob, 2)),
                macro_summary="LLM недоступен — используем базовый сигнал.",
                technical_summary="",
                contradictions=[],
                explanation="Fallback без LLM",
                raw={"fallback": True, "input": raw},
                warnings=[],
            )
        scenario = str(raw.get("scenario") or planned_side).upper()
        prob = raw.get("probability_pct") or raw.get("probability") or 0.0
        try:
            prob = float(prob)
        except Exception:
            prob = base_proba_up * 100
        macro = raw.get("macro_summary") or raw.get("macro") or ""
        technical = raw.get("technical_summary") or raw.get("technical") or ""
        contradictions = [str(x) for x in raw.get("contradictions") or []]
        explanation = raw.get("explanation") or raw.get("rationale") or ""
        return AnalystReport(
            scenario=scenario,
            probability_pct=float(prob),
            macro_summary=str(macro),
            technical_summary=str(technical),
            contradictions=contradictions,
            explanation=str(explanation),
            raw=raw,
            warnings=[],
        )

    def _merge_agent_weights(
        self,
        model_trust: Mapping[str, float] | None,
        agent_weights: Mapping[str, float],
    ) -> Dict[str, float]:
        out = dict(model_trust or {})
        for agent, weight in agent_weights.items():
            key = f"agent:{agent}"
            out[key] = weight
        return out

    def _dynamic_agent_weights(
        self,
        regime_label: str,
        agents: List[str],
    ) -> Dict[str, float]:
        if not agents:
            return {}
        weights = fetch_agent_performance(regime_label)
        if not weights:
            return {}
        out: Dict[str, float] = {}
        for agent in agents:
            agent_weight = weights.get(agent)
            if agent_weight is not None:
                out[agent] = float(agent_weight)
        return out

    def _extract_active_agents(self, context_bundle: Dict[str, Any]) -> List[str]:
        sections = context_bundle.get("sections") or []
        mapping = {
            "новости": "news",
            "macro": "news",
            "макро": "news",
            "ончейн": "onchain",
            "уроки": "lessons",
            "smc": "smc",
            "фибоначчи": "advanced_ta",
            "похожие": "similarity",
        }
        active: set[str] = set()
        for section in sections:
            title = str(section.get("title", "")).lower()
            for key, agent in mapping.items():
                if key in title:
                    active.add(agent)
        return sorted(active)

    def evaluate(
        self,
        base_proba_up: float,
        side: str,
        regime: str,
        lessons: Iterable[dict],
        model_trust: Mapping[str, float] | None = None,
        risk_flags: Iterable[str] | None = None,
        safe_mode: bool = False,
    ) -> ArbiterDecision:
        side_norm = str(side or "LONG").upper()
        base_proba_up = _clamp(base_proba_up)
        base_success = base_proba_up if side_norm == "LONG" else (1.0 - base_proba_up)

        penalty, boost, model_overrides, lesson_notes = self._derive_lesson_adjustments(
            lessons, regime
        )

        trust_adjustment, combined_overrides, trust_notes = self._derive_trust_adjustment(
            model_trust, model_overrides
        )

        success_adj = _clamp(base_success - penalty + boost + trust_adjustment)
        proba_up_adj = success_adj if side_norm == "LONG" else 1.0 - success_adj

        # Determine risk stance
        risk_notes: List[str] = []
        if safe_mode:
            risk_notes.append("safe_mode")
        rf_set = {str(flag) for flag in (risk_flags or [])}
        if any(flag.startswith("block") for flag in rf_set):
            risk_notes.append("risk_flag_block")

        stance = "risk_watch"
        confidence_floor = None
        if success_adj >= self.acceptance_threshold and "risk_flag_block" not in risk_notes:
            stance = "risk_on"
            confidence_floor = 0.62 if success_adj < self.high_bar else 0.72
        elif success_adj < self.acceptance_threshold * 0.75 or safe_mode:
            stance = "risk_off"

        notes = lesson_notes + trust_notes + risk_notes
        if penalty > 0:
            notes.append(f"penalty:{penalty:.2f}")
        if boost > 0:
            notes.append(f"boost:{boost:.2f}")
        if abs(trust_adjustment) > 1e-6:
            notes.append(f"trust_delta:{trust_adjustment:+.2f}")

        return ArbiterDecision(
            proba_up=_clamp(proba_up_adj),
            success_probability=success_adj,
            risk_stance=stance,
            confidence_floor=confidence_floor,
            notes=notes,
            model_overrides=combined_overrides,
        )

    # ------------------------------------------------------------------
    def _derive_lesson_adjustments(
        self, lessons: Iterable[dict], regime: str
    ) -> Tuple[float, float, Dict[str, float], List[str]]:
        total_penalty = 0.0
        total_boost = 0.0
        overrides: Dict[str, float] = {}
        notes: List[str] = []

        for lesson in lessons or []:
            payload = lesson.get("lesson") if isinstance(lesson, dict) else None
            if not isinstance(payload, dict):
                continue
            lesson_regime = str(payload.get("market_regime") or "")
            if lesson_regime and lesson_regime != regime:
                continue
            outcome = float(payload.get("outcome_after") or lesson.get("outcome_after") or 0.0)
            agents = payload.get("involved_agents") or []
            normalized_agents = [_normalize_agent_name(str(a)) for a in agents]

            delta = self.lesson_weight * (abs(outcome) if outcome else 1.0)
            if outcome < 0:
                total_penalty += delta
                for agent in normalized_agents:
                    overrides[agent] = overrides.get(agent, 1.0) - (delta * 0.8)
                if agents:
                    notes.append(
                        f"lesson_penalty:{','.join(sorted(set(normalized_agents)))}"
                    )
            elif outcome > 0:
                total_boost += delta * 0.5
                for agent in normalized_agents:
                    overrides[agent] = overrides.get(agent, 1.0) + (delta * 0.4)
                if agents:
                    notes.append(
                        f"lesson_boost:{','.join(sorted(set(normalized_agents)))}"
                    )

        total_penalty = min(self.max_penalty, total_penalty)
        total_boost = min(self.max_penalty * 0.6, total_boost)
        overrides = {name: max(0.5, min(1.3, value)) for name, value in overrides.items()}

        # Merge with structured lesson weights derived via TrustMonitor (vector search aware)
        extra_weights: Dict[str, float] = {}
        try:
            if self._trust_monitor:
                extra_weights = self._trust_monitor.derive_weights_from_lessons(
                    lessons, regime=regime
                )
        except Exception:
            extra_weights = {}

        for agent, weight in (extra_weights or {}).items():
            base = overrides.get(agent, 1.0)
            merged = base * weight
            overrides[agent] = max(0.5, min(1.4, merged))
            notes.append(f"lesson_weight:{agent}:{weight:.2f}")

        return total_penalty, total_boost, overrides, notes

    # ------------------------------------------------------------------
    def _derive_trust_adjustment(
        self,
        model_trust: Mapping[str, float] | None,
        overrides: Mapping[str, float],
    ) -> Tuple[float, Dict[str, float], List[str]]:
        if not model_trust:
            return 0.0, dict(overrides), []

        per_agent: Dict[str, List[float]] = defaultdict(list)
        for model_name, trust_val in model_trust.items():
            try:
                trust_score = _clamp(float(trust_val))
            except (TypeError, ValueError):
                continue
            agent_key = _normalize_agent_name(model_name)
            per_agent[agent_key].append(trust_score)

        if not per_agent:
            return 0.0, dict(overrides), []

        agent_scores = {
            agent: sum(values) / len(values)
            for agent, values in per_agent.items()
            if values
        }
        if not agent_scores:
            return 0.0, dict(overrides), []

        combined_overrides = dict(overrides)
        trust_notes = []
        multipliers: Dict[str, float] = {}
        for agent, score in agent_scores.items():
            base_multiplier = 0.7 + 0.6 * score  # map [0,1] -> [0.7,1.3]
            merged = base_multiplier * combined_overrides.get(agent, 1.0)
            merged = max(0.5, min(1.5, merged))
            multipliers[agent] = merged
            trust_notes.append(f"trust_{agent}:{score:.2f}")

        combined_overrides.update(multipliers)

        trust_avg = sum(agent_scores.values()) / len(agent_scores)
        trust_adjustment = (trust_avg - 0.5) * (self.trust_weight * 2.0)

        return trust_adjustment, combined_overrides, trust_notes
