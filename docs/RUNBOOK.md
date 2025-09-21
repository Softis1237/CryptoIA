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
   ```

3. Обучите и установите модели динамических индикаторов:
   ```bash
   PYTHONPATH=services/pipeline/src python services/pipeline/ops/train_indicator_params.py \
       --input data/indicator_training.csv \
       --output artifacts/indicator_params \
       --features trend_feature vol_feature news_factor
   ```
   При необходимости замените `data/indicator_training.csv` на свой датасет.

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

## 4. Регрессионные проверки

### 4.1 Pytest
```bash
pytest services/pipeline/tests/test_dynamic_params.py \
       services/pipeline/tests/test_orchestrator_safe_mode.py \
       services/pipeline/tests/test_red_team_agent.py
```

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

