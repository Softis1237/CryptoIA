## Операции и наблюдаемость

Запуск стека:

```
docker compose up -d --build
```

Примечания по конфигу compose:
- Сервисам `pipeline` и `coordinator` задаётся `CCXT_EXCHANGE=binance` (по умолчанию), а также `REDIS_HOST=redis` — кэш/блокировки идут в Redis.

Запуск релиза вручную:

```
docker compose run --rm pipeline python -m pipeline.orchestration.agent_flow --slot=manual
```

Проверка здоровья:

- HTTP эндпоинт `http://localhost:8000/health` возвращает `OK`, если доступны Postgres и S3.
- При `FAIL` проверьте логи контейнеров и настройки подключения.

Метрики Prometheus:
- Укажите `PROM_PUSHGATEWAY_URL`.
- Пушатся: `pipeline_step_seconds`, `pipeline_value` (бизнес/валид./риск метрики).
- Стек: `docker compose up -d pushgateway prometheus grafana`.
- Дашборд: ops/grafana/dashboards/pipeline_overview.json.
- Дашборды: ops/grafana/dashboards/pipeline_overview.json и ops/grafana/dashboards/affiliate_overview.json (поступления/комиссии/конверсия по партнёрам).
- Алерты: ops/prometheus/alerts.yml (Alertmanager на 9093).

Проверка сервиса и LLM

- Быстрый тест пайплайна:
  ```bash
  docker compose run --rm pipeline python -m pipeline.orchestration.agent_flow --slot=manual
  ```
  Ожидаем: личное сообщение от бота (если TELEGRAM_DM_USER_IDS задан и вы нажали /start) + график. Артефакты — в MinIO (bucket `artifacts`).

- Health‑checks (внутри контейнеров):
  - bot/coordinator: http://127.0.0.1:8000/healthz (compose перезапускает контейнер при проблемах)

- Flowise: создайте Chatflow и подставьте `FLOWISE_*_URL` в .env как `http://flowise:3000/api/v1/prediction/<flowId>`. Если endpoints не заданы — LLM‑части пропускаются, остальной конвейер работает.

UI и доступы

- MinIO Console: http://localhost:9001 (логин/пароль как в .env)
- Grafana: http://localhost:3003 (admin/admin) — если порт проброшен в compose
- Prometheus: http://localhost:9094 — если порт проброшен
- Flowise: http://localhost:3002 — если порт проброшен

Частые проблемы

- Telegram InvalidToken: перевыпустите токен в @BotFather (/revoke + /token), пропишите в .env, перезапустите bot, сбросьте вебхук:
  ```bash
  docker compose exec bot python - <<'PY'
  import os
  from telegram import Bot
  b = Bot(os.getenv('TELEGRAM_BOT_TOKEN'))
  print("delete_webhook:", b.delete_webhook(drop_pending_updates=True))
  print("get_me:", b.get_me())
  PY
  ```
- Порты заняты на хосте: измените порты в docker-compose для grafana/prometheus/flowise/pushgateway.
- Нет ключей для on‑chain/LLM: включённые флаги без ключей приведут к пропуску соответствующих блоков с предупреждениями. Остальной пайплайн продолжит работу.

Миграции БД:
- Запускаются при старте контейнера Postgres (если у вас есть init scripts) или применяются вручную:

```
psql $POSTGRES_DSN -f migrations/001_init.sql
psql $POSTGRES_DSN -f migrations/002_pgvector.sql
...
psql $POSTGRES_DSN -f migrations/011_subscriptions_redeem.sql
psql $POSTGRES_DSN -f migrations/012_model_registry.sql
psql $POSTGRES_DSN -f migrations/013_users.sql
psql $POSTGRES_DSN -f migrations/014_news_facts.sql
psql $POSTGRES_DSN -f migrations/015_prediction_outcomes.sql
psql $POSTGRES_DSN -f migrations/016_regime_trust.sql
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

Партнёрская программа (Affiliate):

- Кнопка в боте: «Партнёрка» → выдаёт персональный код и ссылку `https://t.me/<bot>?start=ref_<code>`; партнёры видят свою статистику.
- Начисление: по первой покупке привлечённого пользователя, по умолчанию 50% от суммы (настраивается админом).
- Таблицы: `affiliates`, `referrals`; у пользователей `users.referrer_code/referrer_name`.
- Команды администратора:
  - `/affset <partner_user_id> <percent>` — установить процент партнёру.
  - `/affstats [partner_user_id]` — показать статистику партнёра (если без аргументов, для текущего пользователя).
  - `/affapprove <partner_user_id> [percent] [request_id]` — одобрить заявку и выдать код партнёру; при передаче `request_id` заявка помечается approved.
  - `/affrequests [status]` — показать заявки (по умолчанию pending).
  - `/affmark <request_id> <approved|rejected>` — изменить статус заявки.
