## Агенты и флаги

Общие правила:
- Все агенты регистрируются в `orchestration/agent_flow.py`.
- Флаги окружения позволяют включать/выключать источники и тяжёлые компоненты.

Основные агенты:
- prices (ingest_prices)
- prices_lowtf (ENABLE_DEEP_PRICE)
- news (ingest_news)
- orderbook (ENABLE_ORDERBOOK)
- onchain (ENABLE_ONCHAIN)
- futures (ENABLE_FUTURES)
- social (ENABLE_SOCIAL)
- features — фичи из цен/новостей/OB/ончейн/соц
- regime — ML/эвристика режима
- similar_past — соседи из pgvector
- models_4h / models_12h — прогнозы
- ensemble_4h / ensemble_12h — ансамблирование
- scenarios — LLM/эвристика сценариев
- chart — график цены/уровней
- trade — карточка сделки + verifier

Дополнительные:
- sentiment_quality — качество новостей
- sentiment_index — индекс настроений (news+social)
- anomaly_detection / anomaly_ml — аномалии (эвристика/IForest)
- crisis_detection — кризис (волатильность/imbalance/новости)
- macro_analyzer — макро‑флаги
- alt_data — альт‑данные (pytrends/CME/Coinglass/ликвидность)
- liquidity_analysis — ликвидность (dry/normal)
- contrarian_signal — контртренд по funding + sentiment
- model_trust — подсказка весов по sMAPE
- llm_forecast (ENABLE_LLM_FORECAST) — прогноз LLM
- validation_router / trade_validator / regime_validator / llm_validator — валидация многоуровневая
- models_cv — кросс‑валидация исторических предсказаний
- backtest_validator — простой бэктест по фичам
- risk_estimator — VaR/ES/DD

### MCP mini‑server (минимальный)

- Назначение: единая точка доступа к безопасным tool‑вызовам для LLM.
- Запуск: `python -m pipeline.mcp.server` (env: `MCP_HOST`, `MCP_PORT`).
- Endpoint: `POST /call` с JSON `{"tool": string, "params": object}` => `{ok, result|error}`.
- Доступные инструменты v1:
  - `get_features_tail`: `{features_s3, n=1, columns?}` → последние строки фич.
  - `levels_quantiles`: `{features_s3, qs=[0.2,0.5,0.8]}` → уровни по квантилям close.
  - `news_top`: `{news, top_k=5}` → топ новостей по impact_score.
- Интеграция: LLM‑агенты (debate/scenario/validator) могут вызывать MCP вместо напрямую дергать источники.

### Ансамбль и стэкинг

- По умолчанию ансамбль — взвешенное среднее по inverse-sMAPE.
- Стэкинг (включается `ENABLE_STACKING=1`) обучает линейные веса по зрелым предсказаниям из БД и ценам биржи (ccxt), без внешних ML‑зависимостей.
- Требуется накопить ≥30 наблюдений для устойчивых весов. Иначе используется базовый ансамбль.

### A/B ансамблирование

- Включить `ENABLE_CHALLENGER=1` и выбрать `CHALLENGER_MODE` (`uniform` или `stacking`).
- Результат challenger сохраняется как `agents_predictions` с агентами `ensemble_4h_ch` и `ensemble_12h_ch`.
- Метрики: `pipeline_value{name="ab_div_4h_abs"|"ab_div_12h_abs"}` — абсолютное расхождение y_hat.

### Реестр моделей

### Ретрейны по расписанию

- Скрипт: `python -m pipeline.ops.retrain`.
- Флаги: `ENABLE_RETRAIN=1`, `RETRAIN_INTERVAL_H=24`, `RETRAIN_FEATURES_S3=s3://.../features.parquet`, `RETRAIN_HORIZONS=4h,12h`.
- Планировщик вызывает retrain из `scheduled_runner`.

### KATS/CPD фичи под флагом

### Оптимизация затрат LLM

- Кэш Redis: `ENABLE_LLM_CACHE=1`, `LLM_CACHE_TTL_SEC=600`. Ключ — хэш промпта/модели.
- Бюджет/лимиты: `LLM_CALLS_BUDGET=8` на процесс; при превышении LLM вызовы пропускаются.
- Размер ответа: `LLM_MAX_TOKENS=400`.
- Метрики: публикуются `llm_calls`, `llm_cache_hits`, `llm_failures` (job="llm"). Алерты в `alerts.yml`.

### Контрсигналы/режим

- `contrarian_signal` теперь учитывает режим (label из `regime`) и адаптирует пороги (мягче в тренде).

- KATS: `USE_KATS=1` — добавляет `cp_flag`, `anomaly_flag`, `seasonality_count`.
- CPD fallback без зависимостей: `USE_CPD_SIMPLE=1`, пороги `CPD_Z_THRESHOLD`, окно `CPD_WINDOW` — добавляет `cp_simple_flag`, `cpd_score`.

- Таблица `model_registry` (см. миграцию 011_model_registry.sql) — хранит `name, version, path_s3, params, metrics`.
- Сохранение sklearn‑бандла вызывает реестр: см. `models/models_ml.py:save_models_s3`.


Флаги ключевые:
- USE_COORDINATOR=1 — использовать DAG
- USE_HF_SENTIMENT, HF_SENTIMENT_MODEL
- ENABLE_NEWS_TWITTER/REDDIT/STACKOVERFLOW
- LIQUIDITY_* для порогов ликвидности
- FLOWISE_* для LLM (explain/debate/scenario/validate/forecast)
