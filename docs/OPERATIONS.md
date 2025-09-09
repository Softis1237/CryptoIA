## Операции и наблюдаемость

Запуск стека:

```
docker compose up -d --build
```

Запуск релиза вручную:

```
docker compose run --rm pipeline python -m pipeline.orchestration.agent_flow --slot=manual
```

Метрики Prometheus:
- Укажите `PROM_PUSHGATEWAY_URL`.
- Пушатся: `pipeline_step_seconds`, `pipeline_value` (бизнес/валид./риск метрики).
- Стек: `docker compose up -d pushgateway prometheus grafana`.
- Дашборд: ops/grafana/dashboards/pipeline_overview.json.
- Алерты: ops/prometheus/alerts.yml (Alertmanager на 9093).

Миграции БД:
- Запускаются при старте контейнера Postgres (если у вас есть init scripts) или применяются вручную:

```
psql $POSTGRES_DSN -f migrations/001_init.sql
psql $POSTGRES_DSN -f migrations/002_pgvector.sql
...
```

Feedback aggregator:

```
docker compose run --rm pipeline python -m pipeline.ops.feedback_metrics
```

Ошибки/трассировка:
- SENTRY_DSN — включить Sentry
- JSON_LOGS=1 — структурированные логи loguru

Windmill Flows (рекомендованные):

- Автотренинг ML и регистрация в реестре:

  Файл: `ops/windmill/flows/train_ml_register.py`

  ENV:
  - `FEATURES_S3` (опционально; иначе `s3://$S3_BUCKET/runs/$date/$slot/features.parquet`)
  - `HORIZON_MINUTES` (по умолчанию 240)
  - `MODEL_NAME` (по умолчанию sklearn-bundle)
  - `MODEL_VERSION` (если не задан — timestamp)
  - `ML_MODELS_S3_PREFIX` (куда сохранять bundle)

- Ежедневный сбор метрик обратной связи:

  Файл: `ops/windmill/flows/feedback_metrics_daily.py`

  ENV:
  - `FEEDBACK_WINDOW_DAYS` (по умолчанию 7)

Настройте расписания в Windmill UI (например, nightly для обоих).
