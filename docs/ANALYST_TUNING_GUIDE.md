# Руководство по настройке промптов (V5.0)

Цель — получить устойчивый, качественный Chain-of-Thought и самокритику InvestmentArbiter.

## 1. Подготовка окружения

1. Убедитесь, что применены миграции до `044_arbiter_reasoning.sql` и активированы флаги `ARB_LOG_TO_DB=1`, `ARB_STORE_S3=1` (см. `docs/ENV.md`).
2. Запустите пайплайн в режиме A/B (например, `ARB_ANALYST_AB_PERCENT=25`), чтобы новые промпты постепенно собирали данные.
3. Создайте дашборд по инструкции `docs/MONITORING_ANALYST.md` — метрики `investment-analyst_probability_pct` и `self-critique_probability_delta` помогут отслеживать стабильность.

## 2. Сбор reasoning

Используйте утилиту:

```bash
PYTHONPATH=services/pipeline/src python services/pipeline/ops/export_reasoning.py \
  --limit 40 --mode modern --include-critique --output artifacts/cot_dump.json
```

Поля `analysis` (CoT) и `critique` содержат полные JSON-ответы моделей. Если включено сохранение в S3, оригиналы доступны по пути `runs/<date>/<slot>/arbiter/<run_id>.json`.

## 3. Аналитический цикл

1. Выберите 10–20 запусков на разных рыночных периодах (тренд/флэт, новости, выходы макроданных).
2. Изучите поля:
   - `macro_summary`, `technical_summary`, `contradictions`, `explanation`.
   - `critique.counterarguments`, `critique.invalidators`, `critique.probability_adjustment`.
3. Оцените качество:
   - Учитываются ли новости, ончейн, SMC уровни?
   - Есть ли явные пропуски (например, не замечены сильные противоречия)?
   - Самокритика даёт реальную альтернативу, или повторяет общие слова?
4. Фиксируйте проблемы в issue-трекере (или `docs/ANALYST_PROMPTS.md`) с примерами JSON.

## 4. Изменение промптов

Промпты задаются в `InvestmentArbiter._system_prompt()` и `SelfCritiqueAgent._system_prompt()`. Рекомендации:

- Добавляйте шаги явно: «Сначала опиши макро, затем технический анализ, затем противоречия».
- Включайте ограничения: вероятность >80% только если нет сильных противоречий.
- Приводите примеры: «Если новости бычьи, но цена у уровня 0.618 Фибо → вероятность ≤70%».
- Для SelfCritique просите конкретные invalidate-события и обязательную корректировку вероятности.

После правок:

1. Перезапустите пайплайн (docker/systemd).
2. Соберите новую выборку reasoning по прошлому пункту.
3. Сравните: улучшилось ли качество? Уменьшилось количество «REJECT»? Изменился median `probability_delta`?

## 5. Критерии готовности

- Analyst CoT покрывает все разделы и даёт чёткий прогноз ≤ 300 токенов.
- SelfCritique даёт минимум одно конкретное возражение и корректирует вероятность ≥ |5| п.п. в спорных случаях.
- Метрика `self-critique_probability_delta` в среднем > -8 п.п., но не нулевая (значит, агент действительно спорит).
- Manual review 20 кейсов показывает отсутствие очевидных «слепых пятен».

## 6. Полезные команды

```bash
# последняя reasoning запись в psql
psql "$DATABASE_URL" -c "SELECT run_id, mode, tokens_estimate FROM arbiter_reasoning ORDER BY created_at DESC LIMIT 5;"

# выгрузка контекста определенного запуска
aws s3 cp s3://$S3_BUCKET/runs/2025-01-12/manual/arbiter/20250112T000000Z.json -
```

Зафиксируйте финальные промпты в Git и обновите раздел `docs/TRADING.md`, чтобы команда поддержки знала, как откатиться на старые версии промптов при инцидентах.
