from __future__ import annotations

"""ContextBuilderAgent — собирает оперативную память для инвестиционного анализа."""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..infra import metrics
from ..infra.db import fetch_latest_strategic_verdict
from ..infra.run_lock import slot_lock
from .advanced_ta_agent import AdvancedTAAgent
from .base import AgentResult, BaseAgent
from .lessons import get_relevant_lessons
from .smc_analyst import SMCInput, run as run_smc_v2


class ContextBuilderPayload:
    __slots__ = (
        "run_id",
        "slot",
        "symbol",
        "regime",
        "news",
        "onchain",
        "lessons_context",
        "smc",
        "advanced_ta",
        "neighbors",
        "features_path_s3",
        "force_refresh",
        "tokens_budget",
        "ttl_sec",
    )

    def __init__(
        self,
        *,
        run_id: str,
        slot: str = "context_builder",
        symbol: str = "BTC/USDT",
        regime: Optional[dict] = None,
        news: Optional[Iterable[dict]] = None,
        onchain: Optional[Iterable[dict]] = None,
        lessons_context: Optional[dict] = None,
        smc: Optional[dict] = None,
        advanced_ta: Optional[dict] = None,
        neighbors: Optional[Iterable[dict]] = None,
        features_path_s3: Optional[str] = None,
        force_refresh: bool = False,
        tokens_budget: Optional[int] = None,
        ttl_sec: Optional[int] = None,
    ) -> None:
        self.run_id = run_id
        self.slot = slot
        self.symbol = symbol
        self.regime = regime or {}
        self.news = list(news or [])
        self.onchain = list(onchain or [])
        self.lessons_context = lessons_context or {}
        self.smc = smc or {}
        self.advanced_ta = advanced_ta or {}
        self.neighbors = list(neighbors or [])
        self.features_path_s3 = features_path_s3
        self.force_refresh = force_refresh
        self.tokens_budget = tokens_budget or int(os.getenv("CONTEXT_TOKENS_BUDGET", "32000"))
        self.ttl_sec = ttl_sec or int(os.getenv("CONTEXT_CACHE_TTL", "900"))

    def cache_key(self) -> str:
        key = {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "slot": self.slot,
            "regime": self.regime,
            "lessons": self.lessons_context,
            "smc": self.smc,
            "advanced_ta": self.advanced_ta,
        }
        return json.dumps(key, sort_keys=True, ensure_ascii=False)


class _ContextCache:
    def __init__(self) -> None:
        self._local: Dict[str, Tuple[float, dict]] = {}
        try:
            import redis  # type: ignore[import-untyped]

            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=0,
            )
        except Exception:
            self._redis = None

    def get(self, key: str) -> Optional[dict]:
        if self._redis is not None:
            try:
                raw = self._redis.get(f"ctx:{key}")
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        now = time.time()
        item = self._local.get(key)
        if not item:
            return None
        expires, payload = item
        if now > expires:
            self._local.pop(key, None)
            return None
        return payload

    def set(self, key: str, value: dict, ttl: int) -> None:
        deadline = time.time() + ttl
        self._local[key] = (deadline, value)
        if self._redis is not None:
            try:
                self._redis.setex(f"ctx:{key}", ttl, json.dumps(value, ensure_ascii=False))
            except Exception:
                pass


def _truncate(text: str, max_tokens: int) -> str:
    approx_len = max_tokens * 4
    if len(text) <= approx_len:
        return text
    return text[:approx_len] + "\n…\n[truncated]"


def _format_news(news: List[dict]) -> str:
    lines = []
    for item in news[:10]:
        title = item.get("title") or item.get("headline") or "(без названия)"
        sentiment = item.get("sentiment") or item.get("direction") or "neutral"
        impact = item.get("impact_score") or item.get("magnitude") or 0
        ts = item.get("ts") or item.get("created_at") or ""
        lines.append(f"- {ts} | {sentiment} | impact={impact}: {title}")
    return "\n".join(lines) or "- новостные данные отсутствуют"


def _format_onchain(metrics_payload: List[dict]) -> str:
    if not metrics_payload:
        return "- нет свежих ончейн данных"
    lines = []
    for item in metrics_payload[:10]:
        metric = item.get("metric") or item.get("name")
        val = item.get("value") or item.get("delta")
        desc = item.get("description") or item.get("note")
        lines.append(f"- {metric}: {val} ({desc})" if desc else f"- {metric}: {val}")
    return "\n".join(lines)


def _format_lessons(lessons: List[dict]) -> str:
    if not lessons:
        return "- релевантные уроки не найдены"
    lines = []
    for lesson in lessons[:5]:
        title = lesson.get("title") or "Без названия"
        action = lesson.get("action") or lesson.get("insight") or ""
        risk = lesson.get("risk") or ""
        suffix = f" — {action}" if action else ""
        if risk:
            suffix += f" (риск: {risk})"
        lines.append(f"- {title}{suffix}")
    return "\n".join(lines)


def _format_smc(smc: dict | None) -> str:
    if not smc:
        return "- SMC аналитика недоступна"
    status = smc.get("status") or smc.get("verdict") or "NO_SETUP"
    trend = smc.get("trend_4h") or smc.get("trend")
    zones = smc.get("zones") or {}
    zones_txt = []
    for tf, arr in zones.items():
        zones_txt.append(f"  • {tf}: {len(arr)} зон")
    zones_block = "\n".join(zones_txt) if zones_txt else "  • зоны не найдены"
    return f"- Статус: {status}\n- Тренд: {trend}\n{zones_block}"


def _format_neighbors(neighbors: List[dict]) -> str:
    if not neighbors:
        return "- похожие окна не найдены"
    lines = []
    for item in neighbors[:5]:
        score = item.get("score") or item.get("similarity")
        ts = item.get("ts") or item.get("run_id")
        label = item.get("label") or item.get("regime")
        lines.append(f"- {ts}: score={score}, regime={label}")
    return "\n".join(lines)


@dataclass(slots=True)
class ContextBuilderAgent(BaseAgent):
    name: str = "context-builder"
    priority: int = 18

    def __init__(self) -> None:
        self.name = "context-builder"
        self.priority = 18
        self._cache = _ContextCache()
        self._advanced_ta = AdvancedTAAgent()

    def run(self, payload: dict) -> AgentResult:
        params = ContextBuilderPayload(**payload)
        cache_key = params.cache_key()
        if not params.force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return AgentResult(name=self.name, ok=True, output={"context": cached, "cached": True})

        with slot_lock(f"{self.name}:{params.symbol}"):
            start = time.perf_counter()
            bundle = self._build_bundle(params)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._cache.set(cache_key, bundle, params.ttl_sec)
            tokens_estimate = bundle.get("tokens_estimate", 0.0)
            try:
                metrics.push_values(
                    job=self.name,
                    values={
                        "tokens": float(tokens_estimate),
                        "latency_ms": float(elapsed_ms),
                    },
                    labels={"symbol": params.symbol, "slot": params.slot},
                )
            except Exception:
                pass
        return AgentResult(name=self.name, ok=True, output={"context": bundle, "cached": False})

    def _build_bundle(self, params: ContextBuilderPayload) -> dict:
        sections: List[dict] = []

        regime_label = params.regime.get("label") or params.regime.get("regime") or "unknown"
        regime_conf = params.regime.get("confidence")
        sections.append(
            {
                "title": "Рыночный режим",
                "body": f"Метка: {regime_label}, уверенность={regime_conf}",
            }
        )

        if params.news:
            sections.append({"title": "Новости и макро", "body": _format_news(params.news)})

        if params.onchain:
            sections.append({"title": "Ончейн метрики", "body": _format_onchain(params.onchain)})

        lessons = get_relevant_lessons(params.lessons_context or {}, k=5)
        sections.append({"title": "Уроки из прошлого", "body": _format_lessons(lessons)})

        smc_block = params.smc
        if (
            not smc_block
            and params.features_path_s3
            and os.getenv("CONTEXT_FETCH_SMC", "0") in {"1", "true", "True"}
        ):
            try:
                smc_block = run_smc_v2(SMCInput(run_id=params.run_id))
            except Exception:
                smc_block = {}
        sections.append({"title": "SMC обзор", "body": _format_smc(smc_block)})

        ta_block = params.advanced_ta
        if (
            not ta_block
            and params.features_path_s3
            and os.getenv("CONTEXT_FETCH_TA", "1") in {"1", "true", "True"}
        ):
            try:
                ta_block = self._advanced_ta.run(
                    {
                        "features_path_s3": params.features_path_s3,
                        "symbol": params.symbol,
                        "timeframe": "4h",
                        "lookback": 120,
                    }
                ).output
            except Exception:
                ta_block = {}
        if ta_block:
            fb_support = ta_block.get("fib_support") or []
            fb_resistance = ta_block.get("fib_resistance") or []
            sections.append(
                {
                    "title": "Фибоначчи и уровни",
                    "body": (
                        f"Базовый свинг: high={ta_block.get('swing_high')} low={ta_block.get('swing_low')} bias={ta_block.get('bias')}\n"
                        f"Support: {fb_support}\nResistance: {fb_resistance}"
                    ),
                }
            )

        if params.neighbors:
            sections.append({"title": "Похожие окна", "body": _format_neighbors(params.neighbors)})

        verdict = fetch_latest_strategic_verdict("StrategicData", params.symbol) or {}
        if verdict:
            sections.append({"title": "Стратегические сигналы", "body": json.dumps(verdict, ensure_ascii=False, indent=2)})

        text_parts = [f"# Контекст для {params.symbol}", f"Run: {params.run_id}"]
        for section in sections:
            text_parts.append(f"\n## {section['title']}\n{section['body']}")
        combined_text = _truncate("\n".join(text_parts), params.tokens_budget)
        tokens_estimate = round(len(combined_text) / 4.0, 2)
        return {
            "summary": sections[0],
            "sections": sections,
            "text": combined_text,
            "tokens_estimate": tokens_estimate,
            "built_at": time.time(),
        }


def main() -> None:  # pragma: no cover
    agent = ContextBuilderAgent()
    res = agent.run({
        "run_id": "demo",
        "symbol": "BTC/USDT",
        "regime": {"label": "trend_up", "confidence": 0.62},
        "news": [{"title": "ETF approved", "sentiment": "positive", "impact_score": 0.9, "ts": "2024-05-01"}],
        "lessons_context": {"regime": "trend_up"},
    })
    print(res.output)


if __name__ == "__main__":
    main()
