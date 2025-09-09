# Crypto Forecast Agent Network (MVP)

Полная документация по v2: см. папку `docs/` (старт — `docs/README.md`).

Телеграм‑бот с агентской сетью прогнозов BTC: два релиза в день (00:00 и 12:00, Asia/Jerusalem), краткие объяснения, новости и карточка сделки. Этот репозиторий — стартовая заготовка (MVP): Docker Compose, БД/хранилища, Python‑пайплайн и первый шаг `ingest_prices`.

## Быстрый старт (локально)

1) Скопируйте `.env.example` в `.env` и при необходимости измените переменные.

2) Запустите инфраструктуру:

```
docker compose up -d --build
```

3) Выполните первый прогон модуля цен (пример):

```
docker compose run --rm pipeline \
  python -m pipeline.data.ingest_prices '{"run_id":"2025-09-06-12","slot":"12:00","symbols":["BTCUSDT"],"start_ts":1693939200,"end_ts":1693950000}'
```

Данные сохранятся в MinIO (bucket `artifacts`) по схеме `/runs/{YYYY-MM-DD}/{slot}/<artifact>.parquet` (например, `prices.parquet`).

4) Новости (CryptoPanic + NewsAPI, TextBlob/NLTK):

```
docker compose run --rm pipeline \
  python -m pipeline.data.ingest_news '{"run_id":"2025-09-06-12","slot":"12:00","time_window_hours":12,"query":"bitcoin OR BTC"}'
```

Требуются ключи: `CRYPTOPANIC_TOKEN`, `NEWSAPI_KEY` в `.env`.

5) Telegram Payments бот (по желанию):

В `.env` задайте `TELEGRAM_BOT_TOKEN` и `TELEGRAM_PROVIDER_TOKEN` (и опц. `TELEGRAM_PRIVATE_CHANNEL_ID`). Затем:

```
docker compose up -d bot
```

Команды: `/start`, `/buy`, `/status`. После оплаты бот пришлёт инвайт в приват‑канал (если задан и бот — админ).

- Управление подписками:
  - `/link` — получить инвайт в закрытый канал (для активной подписки)
  - `/renew` — продлить подписку (invoice)
  - `/admin_sweep` — (для OWNER) выгрузить истёкшие подписки из канала
  - Windmill flow: `ops/windmill/flows/subscriptions_sweep.py` — ночной запуск

6) Paper trading (симулятор):

- Разовый запуск исполнителя из последней рекомендации:

```
docker compose run --rm pipeline python -m pipeline.trading.paper_trading executor
# или запустить как отдельный сервис:
docker compose up -d paper_exec
```

- Риск‑луп (SL/TP, каждые 60s):

```
docker compose up -d paper_risk
```

- Закрытие позиций по горизонту и отчёт администратору:

```
docker compose up -d paper_settler
docker compose run --rm pipeline python -m pipeline.trading.paper_trading admin_report
```

7) Flowise (агентные графы) и Windmill (оркестрация):

```
docker compose up -d flowise windmill
```

- Flowise UI: http://localhost:3000 (логин admin/admin)
- Windmill UI: http://localhost:8000 (подключение к локальному Postgres)
  - Health‑flow: `ops/windmill/flows/flowise_health.py` (проверяет `FLOWISE_BASE_URL/api/v1/ping`)
  - Таймауты/ретраи для Flowise: настраиваются через ENV `FLOWISE_TIMEOUT_SEC`, `FLOWISE_MAX_RETRIES`, `FLOWISE_BACKOFF_SEC`

- Примеры flow‑скриптов для Windmill: `ops/windmill/flows/`
  - `predict_release.py` — запускает релиз (через прямой импорт или `docker compose run pipeline`)
  - `prewarm_features.py` — прогрев фичей на 72h
  - `healthcheck_nightly.py` — проверка связности DB и S3
  - Рекомендуемые расписания:
    - predict_release: `0 0,12 * * *` (TZ Asia/Jerusalem)
    - prewarm_features: `0 11,23 * * *`
    - healthcheck_nightly: `0 3 * * *`

8) Observability (опционально):

- Задайте `PROM_PUSHGATEWAY_URL` в `.env` — пайплайн будет пушить длительности шагов (`predict_release`) в Pushgateway.
- Задайте `SENTRY_DSN` — для трекинга ошибок (инициализацию можно добавить в `predict_release` при необходимости).
- Локально можно поднять Pushgateway: `docker compose up -d pushgateway`, и указать `PROM_PUSHGATEWAY_URL=http://pushgateway:9091`.

- Полный стек локальной наблюдаемости:

```
docker compose up -d pushgateway prometheus grafana
```

- Grafana: http://localhost:3001 (admin/admin), дашборд Pipeline Overview.

## Структура

- `docker-compose.yml` — инфраструктура (Postgres, Redis, MinIO, pipeline).
- `migrations/` — SQL‑схемы, автоматически применяются при первом старте Postgres.
- `services/pipeline/src/pipeline` — Python‑сервис, разложен по категориям:
  - `infra/` — конфиг, БД/pgvector, S3/MinIO, метрики, Sentry, редис‑локи, healthcheck.
  - `data/` — сбор данных: `ingest_prices`, `ingest_news`.
  - `features/` — генерация фичей: `features_calc`, `features_kats`.
  - `models/` — модели и тюнинг: `models`, `models_darts`, `models_neuralprophet`, `models_prophet`, `tuning`.
  - `ensemble/` — ансамблирование: `ensemble_rank`.
  - `regime/` — режим рынка: `regime_detect`.
  - `similarity/` — похожие окна (pgvector): `similar_past`.
  - `scenarios/` — сценарии и хелперы LLM: `scenario_modeler`, `scenario_helper`.
  - `reasoning/` — объяснения/арбитраж и схемы JSON: `explain`, `debate_arbiter`, `llm`, `schemas`.
  - `trading/` — торговые компоненты: `trade_recommend`, `verifier`, `publish_telegram`, `paper_trading`, `payments_bot`, `subscriptions`.
  - `reporting/` — графики и отчёты релиза: `charts`, `release_report`.
  - `orchestration/` — запуск релиза: `predict_release`.
  - `utils/` — утилиты: `utils_text`, `calibration`.
  - `ops/prometheus`, `ops/grafana` — конфиги Prometheus и Grafana (дашборд Pipeline Overview).
  - `ops/windmill/flows` — скрипты для Windmill (predict/prewarm/health/подписки/отчёты/tuning).
- `.github/workflows/ci.yaml` — линтеры/проверки.
- `.pre-commit-config.yaml` — ruff/black/isort/mypy.


