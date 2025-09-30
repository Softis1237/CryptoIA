# CryptoIA v2.0 Runbook

Этот документ описывает последовательность действий для подготовки продового окружения, запуска основных сервисов и контроля качества.

## 1. Подготовка окружения

1. Скопируйте `.env.example` в `.env` и заполните продовые значения:
   ```bash
   cp .env.example .env
   ````
   Обязательно задайте:
   - `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_MODEL_MASTER`, `OPENAI_MODEL_VISION`.
   - `SAFE_MODE_RISK_PER_TRADE`, `SAFE_MODE_LEVERAGE_CAP`, `STRATEGIC_DATA_KEYWORDS`, `PROJECT_TASKS_WEBHOOK`.
   - Доступ к PostgreSQL/Redis/S3.

2. Примените миграции:
   ```bash
   alembic upgrade head
   # либо вручную через psql:
   psql "$DATABASE_URL" -f migrations/041_data_sources.sql
   psql "$DATABASE_URL" -f migrations/042_structured_memory.sql
   psql "$DATABASE_URL" -f migrations/043_orchestrator_events.sql
   psql "$DATABASE_URL" -f migrations/044_arbiter_reasoning.sql
   ```

3. Обучите и установите модели динамических индикаторов:
   ```bash
   PYTHONPATH=services/pipeline/src python services/pipeline/ops/train_indicator_params.py \
       --input data/indicator_training.csv \
       --output artifacts/indicator_params \
       --features trend_feature vol_feature news_factor
   ```
   При необходимости замените `data/indicator_training.csv` на свой датасет.

   > Примечание. Датасет можно оперативно сформировать из `data/btc-usd-max.csv`:
   > ```bash
   > PYTHONPATH=services/pipeline/src python -m pipeline.ops.prepare_indicator_data
   > ```
   > Скрипт соберёт `trend_feature`, `vol_feature`, `news_factor` и целевые окна для всех режимов рынка.

4. Интеграция `PROJECT_TASKS_WEBHOOK`

   - Для smoke-теста используйте локальный ресивер:
     ```bash
     PYTHONPATH=services/pipeline/src python ops/scripts/test_project_webhook.py
     ```
     Скрипт подменяет БД, запускает агент `StrategicDataAgent` и выводит JSON с телом POST-запроса.

   - **Linear**: создайте *Automation → External webhook*, скопируйте URL и сохраните в `.env` как `PROJECT_TASKS_WEBHOOK`. Linear принимает запросы без дополнительных заголовков, если webhook создан именно как Automation.

   - **Jira Cloud**: используйте Automation → "Send web request" и выберите Public URL (без авторизации). Если требуется защита, подключите промежуточный relay (n8n/Zapier) и передавайте туда `PROJECT_TASKS_WEBHOOK`.

   - После деплоя перезапустите пайплайн (docker/systemd). Проверяйте новые задачи в выбранной PMS и лог агента `strategic-data-agent`.

## 2. Сервис событий оркестратора

Поднимите фоновый слушатель, чтобы события `data_anomaly` и `post_mortem_feedback` обрабатывались мгновенно:
```bash
PYTHONPATH=services/pipeline/src python services/pipeline/src/pipeline/orchestration/event_listener.py
```
Рекомендуется запускать через systemd/docker compose. Пример unit-файла:
```ini
[Unit]
Description=CryptoIA Orchestrator Event Listener
After=network.target

[Service]
EnvironmentFile=/etc/cryptoia/.env
WorkingDirectory=/opt/cryptoia
ExecStart=/usr/bin/python -m pipeline.orchestration.event_listener
Restart=always

[Install]
WantedBy=multi-user.target
```

## 3. Проверка safe-mode

1. Создайте событие аномалии (SQL):
   ```sql
   INSERT INTO orchestrator_events(event_type, payload)
   VALUES ('data_anomaly', '{"sources": ["mock-source"], "run_id": "manual-test"}');
   ```
2. Запустите orchestrator:
   ```bash
   PYTHONPATH=services/pipeline/src python -m pipeline.orchestration.master_orchestrator_agent
   ```
3. Убедитесь, что в выводе присутствует `"safe_mode": true` и `risk_overrides` с пониженными значениями.
о `pytest` 
## 4. Регрессионные проверки

### 4.1 Pytest
```bash
pytest services/pipeline/tests/test_dynamic_params.py \
       services/pipeline/tests/test_orchestrator_safe_mode.py \
       services/pipeline/tests/test_red_team_agent.py
```

### 4.2 Аналитический совет (CoT)

- `ARB_ANALYST_ENABLED=0` — полный откат к legacy-арбитер. Используйте при инцидентах с LLM.
- `ARB_ANALYST_AB_PERCENT=50` — раскат цепочки рассуждений на 50% запусков (по run_id).
- Метрики Prometheus: `context_builder_tokens`, `context_builder_latency_ms`, `investment-analyst_probability_pct`, `self-critique_probability_delta`.
- Координаты логов (при включённых миграциях v3): таблицы `arbiter_reasoning`, `arbiter_selfcritique`, S3 `runs/<date>/<slot>/arbiter/`.
- Памятки: промпт-тюнинг — `docs/ANALYST_TUNING_GUIDE.md`, контроль LLM — `docs/LLM_FAILSAFE.md`.

### 4.3 Бэктест и бумажная торговля

- Полная инструкция: `docs/BACKTEST_RUNBOOK.md`.
- Paper trading: `docs/PAPER_TRADING_RUNBOOK.md`.
- Live-cutover: `docs/LIVE_TRADING_CHECKLIST.md`.

### 4.2 Бэктест Красной Команды
```bash
PYTHONPATH=services/pipeline/src python - <<'PY'
from pipeline.agents.red_team_agent import RedTeamAgent
from pipeline.agents.memory_guardian_agent import MemoryGuardianAgent

agent = RedTeamAgent(guardian=MemoryGuardianAgent())
# Используйте уроки из БД; для синтетического прогона можно подставить mock-уроки.
print(agent.run({"run_id": "baseline"}).output)
PY
```
Зафиксируйте ключевые метрики (total_return, win_rate) в `docs/METRICS_BASELINE.md`.

## 5. Мониторинг

Метрики Prometheus публикуются через `PROM_PUSHGATEWAY_URL`:
- `master_orchestrator_orchestrator_mode` — статус High-Alert.
- `strategic_data_agent_data_source_anomaly` — количество аномалий.
- `chart_vision_agent_chart_vision_bias` — смещение визуального анализа.
- `red_team_agent_red_team_scenarios` — число сценариев в прогоне.

## 6. Цели ТЗ и отслеживание

Сравните baseline-метрики с целями ТЗ:
- Win Rate +15–20% (модули 1–3).
- Повторяющиеся ошибки −75% (модуль 4).
- Период адаптации −50% (модули 1–3).
- Генерация ≥3 прибыльных гипотез (модуль 5).
Записывайте фактические значения в `docs/METRICS_BASELINE.md` и обновляйте по итогам каждой итерации.

## 7. Чек-лист по ТЗ

Для оперативного контроля открытых задач следите за документом `docs/TODO.md`: там перечислены шаги, которые ещё предстоит выполнить (подключение боевых источников, калибровка моделей, мониторинг и т.д.). После выполнения пунктов обновляйте файл, чтобы документация оставалась актуальной.

## 8. Пошаговое выполнение (с бесплатными ресурсами)

### 8.1 Стратегическое управление данными
- **Источники данных**: используйте бесплатные CSV/JSON из [CryptoDataDownload](https://www.cryptodatadownload.com/), [CoinGecko API](https://www.coingecko.com/en/api/documentation) и открытые дашборды Dune (через экспорт CSV). Пример каталога лежит в `data/sources_catalog.json` (можно переопределить через `DATA_SOURCES_CATALOG`).
- **Интеграция**: `strategic/discovery.py` уже подхватывает JSON каталога; достаточно добавить новые записи в файл.
- **Калибровка trust-score**: используйте скрипт
  ```bash
  PYTHONPATH=services/pipeline/src python -m pipeline.ops.calibrate_source_trust \
      --catalog data/sources_catalog.json \
      --output data/source_trust_calibration.csv
  ```
  Файл можно задать через `SOURCE_TRUST_CALIBRATION`.
- **Авто-задачи**: для теста `PROJECT_TASKS_WEBHOOK` используйте бесплатный сервис типа [Webhook.site](https://webhook.site/) или локальный FastAPI listener.

### 8.2 Динамический технический анализ
- **Датасет для обучения**: скачайте бесплатные OHLCV-файлы с CryptoDataDownload и соберите CSV (как в примере `data/indicator_training.csv`).
- **Обучение моделей**: запускайте `pipeline.ops.train_indicator_params` с собранным CSV (уже добавлено в чек-лист). Это не требует платных сервисов.
- **Валидация Chart Vision**: сформируйте изображения графиков локально (`matplotlib`/`plotly`) и используйте бесплатный tier OpenAI (или локальные модели вроде [LLaVA](https://github.com/haotian-liu/LLaVA)) — главное, чтобы модель могла работать без доп. затрат.
- **Technical Synthesis статистика**: прогоните backtest на локальных данных (см. `pipeline/trading/backtest/runner.py`) и сохраните результаты в `docs/METRICS_BASELINE.md`.
- **Горизонты прогноза**: релизный пайплайн обучает и публикует ансамбли для 30m/4h/12h/24h. Обновляйте ONNX/PKL через `train_indicator_params`, чтобы все горизонты оставались в актуальном состоянии.

- **Event Listener**: запустите локально `python -m pipeline.orchestration.event_listener` (в docker-compose или systemd). Все зависимости бесплатны.
- **Экономический календарь**: используйте бесплатные RSS/JSON, например [Investing.com Economic Calendar (CSV)](https://www.investing.com/economic-calendar/) или [FRED API](https://fred.stlouisfed.org/docs/api/fred/). Экспортируйте данные в `data/eco_calendar.csv` и модифицируйте `eco_calendar.py`, чтобы брать данные из файла при отсутствии API-ключей.
- **Гарантированные релизы в 00:00 и 12:00**: `scheduled_runner` в режиме `RUN_TWICE_DAILY=1` (значение по умолчанию) вычисляет окно до ближайшего слота и всегда запускает Master Orchestrator/Agent flow как минимум дважды в сутки. Даже при переключении на «умную» оркестрацию соблюдается базовый график.
- **Основной пайплайн**: держите `USE_COORDINATOR=1` и при необходимости `USE_MASTER_AGENT=1`, чтобы orchestrator делегировал прогнозы в MasterAgent (`run_master_flow`). Скрипт `predict_release.py` оставьте отключённым (`USE_COORDINATOR=0`), только как запасной вариант для ручного запуска без координации.

### 8.4 Память и обучение
- **Кураторский процесс**: создайте Jupyter-ноутбук для анализа таблицы `agent_lessons_structured` и запускайте его локально (pandas + SQLAlchemy).
- **Мониторинг Memory Guardian**: добавьте логирование (`logger.info`) и используйте Prometheus Gauges (уже подключены) для записи latency и количества релевантных уроков.

### 8.5 Красная Команда
- **Сценарии**: для “ложного пробоя” и “взрывной волатильности” используйте локальные функций в `simulations/red_team/scenario.py` (например, с numpy).
- **Отчёты**: записывайте результаты Red Team в SQLite/CSV и просматривайте в Jupyter/Pandas. Все инструменты бесплатные.

### 8.6 Метрики и мониторинг
- **Baseline**: заполните `docs/METRICS_BASELINE.md` после локальных прогонов (pytest, backtests, safe-mode).
- **Prometheus/Grafana**: запустите их через docker-compose с бесплатными образами (официальные open-source).

### 8.7 Runbook
- Следуйте инструкциям из этого файла (не требуется платных сервисов): `.env`, миграции с `psql`, обучение моделей, safe-mode тесты и мониторинг.

## 9. Полный прогон релизного пайплайна

```bash
PYTHONPATH=services/pipeline/src python -m pipeline.orchestration.agent_flow --slot=manual
```
