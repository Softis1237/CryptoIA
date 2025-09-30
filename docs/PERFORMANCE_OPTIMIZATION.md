# Оптимизация производительности

## Профилирование

1. Запустите master flow через `cProfile`:
   ```bash
   PYTHONPATH=services/pipeline/src python -m cProfile -o artifacts/profile_master.prof \
       services/pipeline/src/pipeline/agents/master.py --slot=manual
   ```
   Анализируйте с `snakeviz artifacts/profile_master.prof`.

2. Для отдельных этапов (features_calc, context builder) используйте встроенные таймеры (см. `infra.metrics.push_durations`). Создайте дашборд «pipeline_latency».

3. Запросы к БД: включите `POSTGRES_LOG_STATEMENT=all` (dev) и ищите медленные запросы (`agent_performance`, `arbiter_reasoning`).

## Потенциальные узкие места

- `features_calc`: вычисление объёмных профилей и новостных фактов. Возможные меры: кеширование OHLC, параллелизация через multiprocessing Pool.
- `ContextBuilderAgent`: загрузка SMC/TA → при необходимости отключайте `CONTEXT_FETCH_TA` / `CONTEXT_FETCH_SMC`.
- LLM вызовы: активируйте кеш (`ENABLE_LLM_CACHE=1`), отслеживайте `_LLM_CALLS`, `_LLM_CACHE_HITS`.

## Рекомендации

- Для тяжёлых вычислений добавьте `@timed` (см. `infra.metrics.timed`) вокруг функций.
- Сохраняйте профили (`artifacts/profile_*.prof`) и анализируйте их ежемесячно.
- Перед включением новых признаков запускайте бенчмарк на 100 релизов и фиксируйте T90.
- Для мониторинга точности прогнозов используйте `services/pipeline/ops/forecast_quality_metrics.py` (MAE/SMAPE/DA) и дашборд `forecast_quality`.
