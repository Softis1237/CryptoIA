from __future__ import annotations

"""Simple sanity validators for LLM responses."""

from typing import List

from loguru import logger

from .schemas import ArbiterAnalystResponse, CritiqueResponse


def validate_analyst(report: ArbiterAnalystResponse) -> List[str]:
    warnings: List[str] = []
    if len(report.macro_summary.strip()) < 10:
        warnings.append("macro_summary_too_short")
    if len(report.technical_summary.strip()) < 10:
        warnings.append("technical_summary_too_short")
    if report.scenario == "LONG" and any("resistance" in c.lower() for c in report.contradictions):
        warnings.append("long_vs_resistance")
    if report.scenario == "SHORT" and any("support" in c.lower() for c in report.contradictions):
        warnings.append("short_vs_support")
    if len(report.explanation) > 2000:
        warnings.append("explanation_too_long")
    if warnings:
        logger.warning("Analyst validation warnings: %s", warnings)
    return warnings


def validate_critique(critique: CritiqueResponse) -> List[str]:
    warnings: List[str] = []
    if not critique.counterarguments:
        warnings.append("no_counterarguments")
    if abs(critique.probability_adjustment) < 1.0:
        warnings.append("tiny_adjustment")
    if warnings:
        logger.warning("Critique validation warnings: %s", warnings)
    return warnings
