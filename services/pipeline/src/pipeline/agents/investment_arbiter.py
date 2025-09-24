from __future__ import annotations

"""Investment arbiter — dynamic risk mediator to prevent analytical paralysis.

The arbiter analyses recent structured lessons, current regime, and base
probabilities to adjust the final conviction of trading recommendations.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


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


class InvestmentArbiter:
    """Meta controller for risk acceptance and dynamic weighting."""

    def __init__(
        self,
        acceptance_threshold: float = 0.7,
        high_bar: float = 0.92,
        max_penalty: float = 0.18,
        lesson_weight: float = 0.08,
    ) -> None:
        self.acceptance_threshold = acceptance_threshold
        self.high_bar = high_bar
        self.max_penalty = max_penalty
        self.lesson_weight = lesson_weight

    def evaluate(
        self,
        base_proba_up: float,
        side: str,
        regime: str,
        lessons: Iterable[dict],
        risk_flags: Iterable[str] | None = None,
        safe_mode: bool = False,
    ) -> ArbiterDecision:
        side_norm = str(side or "LONG").upper()
        base_proba_up = _clamp(base_proba_up)
        base_success = base_proba_up if side_norm == "LONG" else (1.0 - base_proba_up)

        penalty, boost, model_overrides, lesson_notes = self._derive_lesson_adjustments(
            lessons, regime
        )

        success_adj = _clamp(base_success - penalty + boost)
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

        notes = lesson_notes + risk_notes
        if penalty > 0:
            notes.append(f"penalty:{penalty:.2f}")
        if boost > 0:
            notes.append(f"boost:{boost:.2f}")

        return ArbiterDecision(
            proba_up=_clamp(proba_up_adj),
            success_probability=success_adj,
            risk_stance=stance,
            confidence_floor=confidence_floor,
            notes=notes,
            model_overrides=model_overrides,
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
        overrides = {
            name: max(0.5, min(1.3, value)) for name, value in overrides.items()
        }
        return total_penalty, total_boost, overrides, notes

