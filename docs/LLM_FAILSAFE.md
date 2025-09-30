# Контроль качества LLM

## Цель
Минимизировать риски «галлюцинаций» и структурных ошибок от LLM-агентов (InvestmentArbiter, SelfCritique, Debate).

## Стандарты

1. Все LLM-ответы должны быть валидированы через Pydantic-схемы (`pipeline.reaso**n**ing.schemas`).
2. Любая ошибка/отказ — fallback на legacy-алгоритм или дефолтный ответ.
3. Логи ошибок пишутся в Prometheus/лог файл с контекстом.

## Реализация

- В `InvestmentArbiter._run_analyst/_run_critique` используется `ArbiterAnalystResponse` / `CritiqueResponse`.
- При ошибке парсинга — возврат `None`, что переводит запуск в legacy-режим без LLM-влияния.
- В `debate`/`multi_debate` (см. `reasoning/debate_arbiter.py`) рекомендуется:
  1. Парсить ответ через Pydantic.
  2. При ошибке — fallback на список аргументов из модели.
- Метрика `llm_failures` увеличивается при каждом исключении (см. `call_openai_json`).
- Добавьте алерт: `increase(llm_failures[15m]) > 3` — сигнал об ухудшении доступности моделей.

## Ручная проверка

```bash
psql "$DATABASE_URL" -c "SELECT run_id, mode FROM arbiter_reasoning ORDER BY created_at DESC LIMIT 5;"
```

Если видите `mode=legacy` чаще обычного — проверяйте логи LLM/API.
