## Операции и наблюдаемость

Подробнее о Telegram-боте см. TELEGRAM_BOT.md.

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

Анализ последних метрик:

```
make analyze ARGS="price data/prices.parquet"
```

Выведет JSON с последними индикаторами цены. Аналогично доступны команды `orderflow`, `supply-demand` и `patterns` для соответствующих данных.


Метрики Prometheus:
- Укажите `PROM_PUSHGATEWAY_URL`.
- Пушатся: `pipeline_step_seconds`, `pipeline_value` (бизнес/валид./риск метрики).
- Стек: `docker compose up -d pushgateway prometheus grafana`.
- Дашборды: `ops/grafana/dashboards/pipeline_overview.json`, `ops/grafana/dashboards/agents_overview.json` и `ops/grafana/dashboards/affiliate_overview.json` (поступления/комиссии/конверсия по партнёрам).
- Алерты: ops/prometheus/alerts.yml (Alertmanager на 9093).

- Дополнительные метрики release_flow: `p_up_30m`, `p_up_30m_cal`, `interval_width_pct_30m`, `p_up_24h`, `p_up_24h_cal`, `interval_width_pct_24h`.

### Импорт дашборда Agents Overview

1. Откройте Grafana → *Dashboards → Import*.
2. Выберите «Upload JSON file» и укажите `ops/grafana/dashboards/agents_overview.json`.
3. В поле Datasource выберите рабочий Prometheus (тот же, что использует `pipeline_overview`).
4. После импорта проверьте панели:
   - `pipeline_value{name="orchestrator_mode",job="master_orchestrator"}`
   - `pipeline_value{name="chart_vision_bias",job="chart_vision_agent"}`
   - `pipeline_value{name="total",job="memory-guardian-curation"}`
   - `pipeline_value{name="red_team_scenarios",job="red_team_agent"}`
5. При необходимости скорректируйте интервалы (`refresh=30s`) из UI Grafana.

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
- Меню партнёрки: кнопки размещены в две колонки: верх — «Стать партнёром»/«Статистика», середина — «Рефералы»/«Дашборд», низ — «Выплата»/«Партнёрка». Внешняя ссылка (если задана) показывается отдельной кнопкой под меню, админ‑кнопка «🛠️» идёт последней.
- Начисление: по первой покупке привлечённого пользователя, по умолчанию 50% от суммы (настраивается админом).
- Таблицы: `affiliates`, `referrals`; у пользователей `users.referrer_code/referrer_name`.
- Команды администратора:
  - `/affset <partner_user_id> <percent>` — установить процент партнёру.
  - `/affstats [partner_user_id]` — показать статистику партнёра (если без аргументов, для текущего пользователя).
  - `/affapprove <partner_user_id> [percent] [request_id]` — одобрить заявку и выдать код партнёру; при передаче `request_id` заявка помечается approved.
  - `/affrequests [status]` — показать заявки (по умолчанию pending).
  - `/affmark <request_id> <approved|rejected>` — изменить статус заявки.

Редактирование контента через бота:

- Кнопки «Редактировать бонусы», «Добавить новость», «Hero A/B» запускают диалог из двух шагов.
- После ввода текста или файла бот запрашивает подтверждение.
- Отмена — кнопка «Отменить» или команда `/cancel`.
- Через 60 с бездействия диалог завершается по тайм‑ауту.

Подробное описание клавиатур, команд и переменных окружения бота приведено в [TELEGRAM_BOT.md](TELEGRAM_BOT.md).

  - `/affmark <request_id> <approved|rejected>` — изменить статус заявки.
