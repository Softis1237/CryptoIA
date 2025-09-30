# CryptoIA Analyst Evolution — Roadmap v3.0

## Vision

Создать аналитический контур, в котором решения принимаются не отдельными агентами, а «советом директоров»: контекст → цепочка рассуждений → самокритика → сделка. Цель – повысить точность сигналов минимум на 20 % и сократить количество ложных входов не менее чем на 15 %.

## Поток данных

```
┌────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│ Data Layer │─────▶│ ContextBuilderAgent  │─────▶│ InvestmentArbiter    │
└────────────┘      │  (оперативная память)│      │  (CoT анализ)        │
                     └─────────┬────────────┘      └─────────┬────────────┘
                               │                             │
                               ▼                             ▼
                     ┌──────────────────────┐      ┌──────────────────────┐
                     │ SelfCritiqueAgent    │─────▶│ Trade / Risk Engine  │
                     └──────────────────────┘      └──────────────────────┘
```

### Источники контекста

| Блок                | Источник                               | Примечание |
|---------------------|----------------------------------------|------------|
| Макро/новости       | KnowledgeCoreAgent, ingest_news        | Сводка, влияние, доверие |
| Рыночный режим      | regime_detect, regime ML               | Метка + вероятность |
| Ончейн              | AlphaHunterAgent                       | Ключевые метрики и аномалии |
| Исторический опыт   | MemoryGuardian / lessons               | 3–5 уроков по контексту |
| SMC уровни          | SMCAnalyst                             | Зоны и статус сетапа |
| Продвинутый ТА      | AdvancedTAAgent                        | Фибоначчи, конвергенции |
| Ретрососеди         | Similar Past                           | Топ‑N окон для ссылки |

### Объекты данных

- **ContextBundle** — итог оперативной памяти. Содержит `sections`, `summary`, `token_estimate`, `source_meta`.
- **AnalystVerdict** — результат CoT (`scenario`, `probability_pct`, `macro_summary`, `technical_notes`, `contradictions`, `explanation`).
- **CritiqueReport** — оппонент (`counterarguments`, `invalidators`, `missing_factors`, `probability_adjustment`, `recommendation`).
- **ArbiterDecision** — объединённый результат: скорректированная вероятность + stance + ссылки на контекст/критика.

### Архитектурные решения

1. **Кеширование**
   - Redis (`context:{run_id}`) с TTL 15 мин., fallback в процесс.
   - Контекст строится детерминированно: одна версия — один ключ.
2. **A/B-контур**
   - `ARB_ANALYST_ENABLED`, `ARB_ANALYST_AB_PERCENT`. По run_id вычисляем bucket.
   - fallback на старый arbiter для контроля.
3. **Мониторинг**
   - Метрики Prometheus: `context_builder_tokens`, `context_builder_latency_ms`, `arbiter_tokens`, `arbiter_probability`, `selfcritique_delta`.
   - Логи CoT/critique в S3 + БД (`arbiter_reasoning`, `arbiter_selfcritique`).
4. **Тестирование**
   - Unit: advanced TA, context builder, arbiter parsing, critique parsing.
   - Integration: smoke run через `predict_release` (feature flag).
   - Offline backtest с фиксацией precision/recall.

### Контрольные точки

1. **Backlog & схемы** — текущий документ, миграции для логов, env-флаги.
2. **ContextBuilderAgent** — сбор данных, кеш, unit тесты.
3. **InvestmentArbiter v2** — CoT промпт, парсер, совместимость с trust/lessons.
4. **SelfCritiqueAgent** — контраргументы, корректировка вероятности.
5. **Мониторинг** — новые Gauge/Histogram, дашборд (tokens/latency/outcomes).
6. **A/B раскат** — включение по слотам, сравнение precision.
7. **Документация/KPI** — обновить README, TRADING, RUNBOOK, ROADMAP; зафиксировать метрики после прогонов.

## Открытые вопросы

- Как измерять «качественность» контекста? Предлагается логировать количество источников и уникальных уроков.
- Нужно ли хранить CoT полностью? Предлагается сохранять JSON + markdown и делать обрезку >20k символов.
- Механизм rollback: env-анфлаг + retention кешей + напоминание в runbook.
- Динамические веса агентов: метрики выигрышей/поражений per regime хранятся в `agent_performance`. См. утилиту `services/pipeline/ops/update_agent_performance.py`.

## Предполагаемые миграции

- Таблица `arbiter_reasoning` (run_id, created_at, context_ref, analysis_json, tokens, mode).
- Таблица `arbiter_selfcritique` (run_id, created_at, recommendation, delta_pct, payload_json).

Миграция `044_arbiter_reasoning.sql` добавлена в репозиторий; выполните её перед включением нового контура.

Эти миграции описаны в backlog и будут добавлены по мере внедрения.
