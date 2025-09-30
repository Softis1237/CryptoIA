# Логирование и трассировка агентов

## Цель
Обеспечить прозрачность работы комплексной многоагентной системы.

## Общий подход

- Включите `LOG_LEVEL=INFO` / `DEBUG` по необходимости, `JSON_LOGS=1` для структурированных логов.
- Используйте `RUN_ID` и `slot` как ключевые поля корреляции.
- Для CoT-агентов используйте таблицы `arbiter_reasoning`, `arbiter_selfcritique`.
- Для пайплайна: `runs/{date}/{slot}/` в S3 содержит графики, reasoning, отчёты.

## По агентам

| Агент | Логи | Метрики |
|-------|------|---------|
| `ContextBuilderAgent` | `context-builder` logger + Redis cache hits | `context_builder_tokens`, `context_builder_latency_ms` |
| `InvestmentArbiter` | Логи `arbiter.*`, reasoning в БД/S3 | `investment-analyst_probability_pct`, `self-critique_probability_delta` |
| `SelfCritiqueAgent` | Ошибки критики (`critique.failed`) | см. выше |
| `AdvancedTAAgent` | `advanced-ta-agent` | `advanced-ta-agent_range`, `close` |
| `MasterAgent` | Основной pipeline log (см. `services/pipeline/src/pipeline/agents/master.py`) | - |

## Трассировка

1. Найдите `run_id` (из БД/Telegram).
2. Посмотрите `arbiter_reasoning` / `arbiter_selfcritique` (CoT).
3. S3 `runs/<date>/<slot>/` — графики, reasoning JSON.
4. Логи агентов (`master`, `context-builder`, `investment-arbiter`).
5. Telegram сообщение (в `bot` и `publisher` логах).

## Troubleshooting

- `ARB_ANALYST_ENABLED=0` — fallback на legacy.
- `ARB_LOG_TO_DB=0`, `ARB_STORE_S3=0` — временно выключить логирование при отладке.
- Трагетируйте ошибки LLM: `increase(llm_failures[15m]) > 3`.

Прозрачность критична: документируйте обнаруженные инциденты в `docs/INCIDENTS_LOG.md`.
