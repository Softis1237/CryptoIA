from __future__ import annotations

import json
import hashlib
import math

import statistics


from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from ..mcp.client import call_tool as _call_tool
from ..infra.db import insert_agent_lesson, insert_agent_lesson_metrics
from ..reasoning.llm import call_openai_json


@dataclass
class MemoryCompressInput:
    n: int = 20  # take last N run summaries
    scope: str = "global"


def _lessons_to_unique_payloads(raw_lessons: Iterable[Any]) -> list[dict[str, str]]:
    """Normalize LLM память в компактные карточки и убираем дубликаты."""
    uniq: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for item in raw_lessons:
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "Lesson")
            insight = str(item.get("insight") or item.get("summary") or item.get("lesson") or "")
            action = str(item.get("action") or item.get("recommendation") or "")
            risk = str(item.get("risk") or item.get("risk_flags") or item.get("context") or "")
        else:
            text = str(item or "").strip()
            if not text:
                continue
            title = text[:80]
            insight = text
            action = ""
            risk = ""

        payload = {
            "title": title.strip() or "Lesson",
            "insight": insight.strip(),
            "action": action.strip(),
            "risk": risk.strip(),
        }
        norm = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        lesson_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if lesson_hash in seen_hashes:
            continue
        seen_hashes.add(lesson_hash)
        uniq.append(payload)
        if len(uniq) >= 7:
            break
    return uniq


def _first_non_empty_text(values: Any) -> str:
    """Return first непустой текст из переданного значения."""

    if isinstance(values, str):
        return values.strip()
    if isinstance(values, IterableABC):
        for val in values:
            s = str(val or "").strip()
            if s:
                return s
    return ""


def _format_probability(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(f):
        return ""
    if 0.0 <= f <= 1.0:
        f *= 100.0
    return f"{f:.1f}%"


def _join_risk_flags(flags: Any) -> str:
    if isinstance(flags, str):
        return flags.strip()
    if isinstance(flags, IterableABC):
        items = [str(it or "").strip() for it in flags]
        items = [it for it in items if it]
        if items:
            return ", ".join(dict.fromkeys(items))
    return ""


def _outcome_summary(outcome: Any) -> Tuple[str, str]:
    """Сформировать сводку и рекомендации по фактическому исходу прогноза."""

    if not isinstance(outcome, dict):
        return "", ""

    fragments: list[str] = []
    actions: list[str] = []

    def _handle_block(label: str, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        pieces: list[str] = []
        direction = payload.get("direction_correct")
        if direction is True:
            pieces.append("направление верное")
        elif direction is False:
            pieces.append("направление ошибочное")
            actions.append(f"Перепроверить сигналы для горизонта {label}")
        err_pct = payload.get("error_pct")
        try:
            if err_pct is not None:
                err_val = abs(float(err_pct))
                pieces.append(f"ошибка {err_val * 100:.2f}%")
                if err_val > 0.015:
                    actions.append(f"Сократить риск для горизонта {label}")
        except (TypeError, ValueError):
            pass
        if pieces:
            fragments.append(f"{label}: {', '.join(pieces)}")

    horizons = outcome.get("horizons") if isinstance(outcome, dict) else None
    if isinstance(horizons, dict):
        for hz, payload in horizons.items():
            _handle_block(str(hz), payload)
    else:
        for key, payload in outcome.items():
            if key in {"run_id", "created_at"}:
                continue
            _handle_block(str(key), payload)

    insight = "; ".join(fragments)
    action = "; ".join(dict.fromkeys(actions)) if actions else ""
    return insight, action



def _fallback_lessons(
    rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """Локальное сжатие памяти, когда LLM недоступна или вернула пустой JSON."""

    lessons: list[dict[str, str]] = []
    stats: dict[str, dict[str, Any]] = {}

def _fallback_lessons(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Локальное сжатие памяти, когда LLM недоступна или вернула пустой JSON."""

    lessons: list[dict[str, str]] = []

    for row in rows:
        final = row.get("final") or {}
        regime = str(final.get("regime") or "").strip()
        slot = str(final.get("slot") or "").strip()
        created_at = str(row.get("created_at") or "").split("T", 1)[0].strip()
        title_parts = [part for part in (regime, slot, created_at) if part]
        if not title_parts and row.get("run_id"):
            title_parts.append(str(row.get("run_id")))
        title = " ".join(" ".join(title_parts).split()[:12]) or "Запуск системы"

        insight_parts: list[str] = []
        if regime:
            insight_parts.append(f"Режим: {regime}")
        e4 = final.get("e4") or {}
        e12 = final.get("e12") or {}
        p4 = _format_probability(e4.get("proba_up"))
        p12 = _format_probability(e12.get("proba_up"))
        if p4 and p12:
            insight_parts.append(f"Вероятность роста 4ч {p4}, 12ч {p12}")
        elif p4:
            insight_parts.append(f"Вероятность роста 4ч {p4}")
        elif p12:
            insight_parts.append(f"Вероятность роста 12ч {p12}")
        ta = final.get("ta")
        if isinstance(ta, dict):
            sentiment = str(ta.get("technical_sentiment") or "").strip()
            if sentiment:
                insight_parts.append(f"Теханализ: {sentiment}")
            key_point = _first_non_empty_text(ta.get("key_observations"))
            if key_point:
                insight_parts.append(key_point)

        outcome_insight, action_hint = _outcome_summary(row.get("outcome"))
        if outcome_insight:
            insight_parts.append(f"Итог: {outcome_insight}")

        insight_text = "; ".join(insight_parts) or "Короткая сводка без детализированных данных."

        action_parts: list[str] = []
        if action_hint:
            action_parts.append(action_hint)
        risk_flags = final.get("risk_flags")
        risk_text = _join_risk_flags(risk_flags)

        risk_from_flags = bool(risk_text)

        if risk_text:
            risk = risk_text
            action_parts.append(f"Контролировать риски: {risk_text}")
        else:
            risk = "Проверить ликвидность и новости перед повторением шага."

        if not action_parts:
            action_parts.append("Сверить запуск со свежими данными рынка и обновить план.")


        lesson = {
            "title": title,
            "insight": insight_text,
            "action": " ".join(dict.fromkeys(action_parts)),
            "risk": risk,
        }
        norm = json.dumps(lesson, ensure_ascii=False, sort_keys=True)
        lesson_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        stats[lesson_hash] = {
            "has_outcome": bool(outcome_insight),
            "risk_from_flags": risk_from_flags,
            "action_from_outcome": bool(action_hint),
            "insight_length": len(lesson["insight"]),
        }
        lessons.append(lesson)
        if len(lessons) >= 5:
            break
    return lessons, stats


def _fallback_quality(
    rows: Iterable[dict[str, Any]],
    lessons: list[dict[str, str]],
    stats: dict[str, dict[str, Any]],
    candidate_count: int,
) -> dict[str, Any]:
    rows_list = list(rows)
    metrics: dict[str, Any] = {
        "rows_processed": len(rows_list),
        "rows_with_outcome": sum(1 for row in rows_list if row.get("outcome")),
        "rows_with_risk_flags": sum(
            1
            for row in rows_list
            if _join_risk_flags((row.get("final") or {}).get("risk_flags"))
        ),
        "lessons_candidates": int(candidate_count),
        "lessons_final": len(lessons),
    }
    if not lessons:
        metrics.update(
            {
                "outcome_coverage": 0.0,
                "risk_flag_ratio": 0.0,
                "action_hint_ratio": 0.0,
                "default_risk_ratio": 1.0,
                "avg_insight_length": 0.0,
                "median_insight_length": 0.0,
                "unique_titles": 0,
            }
        )
        return metrics

    lengths: list[int] = []
    outcome_flags = 0
    risk_flags = 0
    action_hints = 0
    for lesson in lessons:
        norm = json.dumps(lesson, ensure_ascii=False, sort_keys=True)
        lesson_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        info = stats.get(lesson_hash) or {}
        if info.get("has_outcome"):
            outcome_flags += 1
        if info.get("risk_from_flags"):
            risk_flags += 1
        if info.get("action_from_outcome"):
            action_hints += 1
        length = int(info.get("insight_length") or len(lesson.get("insight") or ""))
        lengths.append(length)

    total = len(lessons)
    metrics.update(
        {
            "outcome_coverage": round(outcome_flags / total, 4),
            "risk_flag_ratio": round(risk_flags / total, 4),
            "action_hint_ratio": round(action_hints / total, 4),
            "default_risk_ratio": round(1.0 - (risk_flags / total), 4),
            "avg_insight_length": round(statistics.mean(lengths), 2) if lengths else 0.0,
            "median_insight_length": round(statistics.median(lengths), 2) if lengths else 0.0,
            "unique_titles": len({lesson.get("title") for lesson in lessons if lesson.get("title")}),
        }
    )
    return metrics

        lessons.append(
            {
                "title": title,
                "insight": insight_text,
                "action": " ".join(dict.fromkeys(action_parts)),
                "risk": risk,
            }
        )
        if len(lessons) >= 5:
            break
    return lessons



def run(inp: MemoryCompressInput) -> Dict[str, Any]:
    # Pull recent summaries via MCP tool to reuse existing code path
    items = _call_tool("get_recent_run_summaries", {"n": int(inp.n)}) or {"items": []}
    rows = items.get("items") or []
    if not rows:
        return {"status": "no-data", "inserted": 0, "lessons": []}

    sys = (
        "Ты работаешь в риск-отделе трейдингового фонда. Тебе дают заметки о прошлых запусках системы. "
        "Сожми знания в 3-7 чётких уроков. Каждый урок — объект с ключами title, insight, action, risk. "
        "title: до 12 слов, конкретная тема. insight: что произошло и почему это важно. "
        "action: практический шаг на будущее. risk: ограничение или условие применения. "
        "Верни строго JSON {\"lessons\":[{...}]} без дополнительных полей. Не придумывай точные числа и не повторяйся."
    )
    usr = "Summaries: " + json.dumps(rows, ensure_ascii=False)
    llm_error: Exception | None = None
    try:
        data = call_openai_json(sys, usr)
    except Exception as exc:  # noqa: BLE001
        llm_error = exc
        data = None

    lessons = _lessons_to_unique_payloads(((data or {}).get("lessons") or []))
    mode = "llm"

    metrics_payload: dict[str, Any] | None = None
    metrics_id: int | None = None
    if not lessons:
        fallback_lessons, fallback_stats = _fallback_lessons(rows)
        lessons = _lessons_to_unique_payloads(fallback_lessons)
        mode = "fallback"
        metrics_payload = _fallback_quality(rows, lessons, fallback_stats, len(fallback_lessons))
        try:
            metrics_id = insert_agent_lesson_metrics(
                scope=inp.scope,
                mode=mode,
                rows_processed=metrics_payload["rows_processed"],
                lessons_inserted=len(lessons),
                metrics=metrics_payload,
            )
        except Exception:
            metrics_id = None

    if not lessons:
        lessons = _lessons_to_unique_payloads(_fallback_lessons(rows))
        mode = "fallback"


    inserted = 0
    stored: list[dict[str, Any]] = []
    for ls in lessons:
        try:

            norm = json.dumps(ls, ensure_ascii=False, sort_keys=True)
            lesson_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            meta_payload: dict[str, Any] = {
                "source": "compressor",
                "n": inp.n,
                "hash": lesson_hash,
                "mode": mode,
                "llm_error": str(llm_error) if llm_error and mode == "fallback" else None,
            }
            if mode == "fallback":
                if metrics_id:
                    meta_payload["quality_metrics_id"] = metrics_id
                if metrics_payload:
                    meta_payload["quality_snapshot"] = metrics_payload
            insert_agent_lesson(ls, scope=inp.scope, meta=meta_payload)

            insert_agent_lesson(
                json.dumps(ls, ensure_ascii=False),
                scope=inp.scope,
                meta={
                    "source": "compressor",
                    "n": inp.n,
                    "hash": hashlib.sha256(json.dumps(ls, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
                    "mode": mode,
                    "llm_error": str(llm_error) if llm_error and mode == "fallback" else None,
                },
            )

            inserted += 1
            stored.append(ls)
        except Exception:
            continue

    return {
        "status": "ok",
        "inserted": inserted,
        "lessons": stored,
        "mode": mode,
        "metrics": metrics_payload,
    }

    return {"status": "ok", "inserted": inserted, "lessons": stored, "mode": mode}



def main():
    import sys, json
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.agents.memory_compressor '<json_payload>'", file=sys.stderr)
        sys.exit(2)
    payload = MemoryCompressInput(**json.loads(sys.argv[1]))
    res = run(payload)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
