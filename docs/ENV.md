ENV_FULL.md — подробная сводка переменных окружения

Этот файл агрегирует все переменные окружения, используемые в CryptoIA. В большинстве случаев необязательные компоненты отключены по умолчанию, и вы можете включить их, установив значение в 1 или присвоив API‑ключ.

1. Основная инфраструктура

POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT  — параметры подключения к PostgreSQL. Используются всеми модулями для сохранения результатов, сигналов, прогнозов и торговых позиций.

REDIS_HOST, REDIS_PORT — адрес кеша Redis для промежуточных данных, кэша LLM и блокировки операций.

S3_ENDPOINT_URL, S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET — настройки MinIO/S3. Сюда записываются parquet‑файлы с ценами и признаками, графики и модели.

TIMEZONE — таймзона слотов. По умолчанию Asia/Jerusalem. Используется для округления времени при публикации и расписания задач.

2. Источники данных и toggles

CRYPTOPANIC_TOKEN, NEWSAPI_KEY — токены для агрегаторов новостей CryptoPanic и NewsAPI.

RSS_NEWS_SOURCES — необязательный список RSS‑лент (через запятую/пробел/перенос строки). Если платные ключи не заданы, новости подтягиваются из RSS (CoinDesk, Cointelegraph, Bitcoin Magazine и др.) через aiohttp+feedparser. Можно также переопределить COINDESK_RSS_URL/COINTELEGRAPH_RSS_URL.

RSS_SOURCES_FILE — путь к файлу со списком RSS‑источников (по одному URL в строке). Если не задан, используется встроенный список `pipeline/data/rss_sources_full.txt` (при наличии).

RSS_TIMEOUT_SEC — таймаут загрузки одной RSS‑ленты (по умолчанию 10 с).

RSS_CONCURRENCY — ограничение одновременных запросов RSS (по умолчанию 8).

Аккаунты и ключи (где взять)

- Telegram Bot Token — @BotFather → /newbot → токен; если токен скомпрометирован или отклоняется, сделайте /revoke и /token (новый токен).
- OpenAI API Key — https://platform.openai.com → Create API key → OPENAI_API_KEY в .env.
- Dune API Key — https://dune.com → профиль → API Keys → DUNE_API_KEY. Создайте/найдите публичные SQL‑запросы и укажите их ID в переменных DUNE_QUERY_*.
- CryptoPanic / NewsAPI — по желанию. Без них новости берутся из RSS (уже настроено).
- Flowise — работает локально (docker), UI доступен на http://localhost:3002 (если проброшен порт). Для каждого сценария создайте Chatflow и укажите его endpoint в .env.

Discovery (PatternDiscoveryAgent)

- Параметры по умолчанию задаются через переменные окружения:
  -  — символ для анализа (по умолчанию BTC/USDT)
  -  — биржа ccxt (binance и т.п.)
  -  — исторический горизонт в днях для таймфрейма 1h (например, 120)
  -  — порог крупного движения (например, 0.05 → 5%)
  -  — ширина окна для оценки будущего движения (например, 24)
  -  — окно до события для сводки контекста (например, 48)
  -  — ограничение числа событий за прогон (например, 40)
  -  — 1/0: dry-run без записи в БД 
  - Отдельно для 4h: , , , , 

Altdata (ликвидность/проскальзывание)

-  — список бирж через запятую (binance,bybit,okx,kraken)
-  — размер рыночного ордера в USD для оценки проскальзывания (по умолчанию 100000)
  - Модуль  пишет метрики: , , а также 

MCP (инструменты)

- Примеры запросов:
  - : 
  - : 
  - : 
  - : 
  - : 

INSIGHTS_WINDOW_H — окно для учёта пользовательских инсайтов в новостях (по умолчанию 24ч).
INSIGHTS_MIN_TRUTH — порог оценки правдивости для учёта инсайта (0..1, по умолчанию 0.6).
INSIGHTS_MIN_FRESH — порог актуальности для учёта инсайта (0..1, по умолчанию 0.5).
INSIGHTS_MAX — максимум инсайтов, добавляемых в один запуск (по умолчанию 30).

ENABLE_ORDERBOOK (0/1) — загрузка стакана (orderbook). Требует Binance/Bybit/etc через ccxt.

ENABLE_ONCHAIN, DUNE_API_KEY — включение ончейн‑метрик и ключ Dune. Если ключа нет, рекомендуем оставить ENABLE_ONCHAIN=0, чтобы не показывать «on‑chain: 0 сигналов».

Переменные DUNE_QUERY_* — ID запросов для метрик: ACTIVE_ADDRESSES, EXCHANGES_NETFLOW_SUM, MVRV_Z_SCORE, SOPR, MINERS_BALANCE_SUM, TRANSFERS_VOLUME_SUM.

ENABLE_FUTURES, FUTURES_PROVIDER — сбор funding rate, mark price и open interest (default binance).

CCXT_EXCHANGE — имя биржи в ccxt для цен и стакана (по умолчанию binance). Альтернативно поддерживается CCXT_PROVIDER. Дополнительно:

- CCXT_TIMEOUT_MS — таймаут запросов ccxt (по умолчанию 20000 мс)
- CCXT_RETRIES — число ретраев на сетевые ошибки (по умолчанию 3)
- CCXT_RETRY_BACKOFF_SEC — базовый бэкофф между ретраями (по умолчанию 1.0с)

ENABLE_SOCIAL, SOCIAL_QUERY, SOCIAL_SUBREDDITS, SOCIAL_MAX_ITEMS, TWITTER_BEARER_TOKEN, REDDIT_USER_AGENT — загрузка твитов/реддита, фильтры и лимиты.

ENABLE_DEEP_PRICE — загрузка свечей 1s/5s.

 

USE_HF_SENTIMENT, HF_SENTIMENT_MODEL — использовать HuggingFace модель для оценки настроений (по умолчанию — off). Модель по умолчанию: nlptown/bert-base-multilingual-uncased-sentiment.

USE_SOCIAL_EMB, SOCIAL_EMB_MODEL — векторизация текстов соц‑сетей с помощью sentence‑transformers.

ENABLE_ALT_DATA — сбор альтернативных данных (Google Trends, CME OI, опционы, ликвидность). Для работы потребуются:

QUANDL_API_KEY (CME OI),

COINGLASS_API_KEY (опционные данные),

ALT_GOOGLE_QUERIES — запросы для Google Trends (по умолчанию bitcoin,btc price),

ALT_EXCHANGES — список бирж для анализа ликвидности (binance,bybit,okx,kraken),

ALT_ORDER_SIZE_USD, LIQUIDITY_MIN_USD, LIQUIDITY_MAX_SLIP_BPS — параметры расчёта глубины рынка,

ALT_SECURITY_RSS, ALT_REGULATORY_RSS — RSS‑ленты для мониторинга инцидентов и регулирования.

3. LLM и Flowise

OPENAI_API_KEY, OPENAI_MODEL — ключ и модель для OpenAI (по умолчанию gpt-4o-mini). Без ключа вызовы LLM пропускаются.

LLM_MAX_TOKENS — максимальное количество токенов в ответе (4096 по умолчанию — достаточно «пространства для мысли» без принудительного длинного ответа).

ENABLE_LLM_CACHE, LLM_CACHE_TTL_SEC — включение и время жизни кэша для LLM ответов (по умолчанию 600 с). Использует Redis.

LLM_CALLS_BUDGET — лимит вызовов LLM за процесс (8 по умолчанию).

FLOWISE_BASE_URL — базовый URL сервиса Flowise. Отдельные endpoints:

FLOWISE_EXPLAIN_URL, FLOWISE_DEBATE_URL, FLOWISE_SCENARIO_URL — конечные точки для генерации объяснения, дебатов и сценариев.
ENABLE_NEWS_SENTIMENT (0/1) — включает LLM‑оценку тональности/импакта для ingest_news.
FLOWISE_SENTIMENT_URL — endpoint Flowise для оценки новостей (ожидается JSON‑список объектов с полями sentiment/impact_score/topics).

FLOWISE_VALIDATE_URL — валидация торговых рекомендаций (если ENABLE_LLM_VALIDATOR=1).

FLOWISE_TIMEOUT_SEC, FLOWISE_MAX_RETRIES, FLOWISE_BACKOFF_SEC — таймаут, число повторов и коэффициент экспоненциальной задержки для запросов к Flowise.

ENABLE_LLM_FORECAST, FLOWISE_FORECAST_URL — опциональный LLM‑прогноз (флаг и URL).

4. Весовые и режимные параметры

ENABLE_REGIME_WEIGHTING — включает перерасчёт весов ансамбля в зависимости от режима рынка.

ALPHA_UPDATE_INTERVAL_H, REGIME_ALPHA_MIN, REGIME_ALPHA_MAX, REGIME_ALPHA_DECAY_DAYS, REGIME_ALPHA_SMOOTH_BETA — параметры адаптации весов (см. алгоритм в regime/).

NEWS_CTX_COEFF_A/B/C, NEWS_CTX_WIN_SHORT/LONG_MIN, NEWS_CTX_K — параметры шкалирования контекстного индекса новостей (используются в features_calc).

NEWS_FACTS_TOPK, NEWS_FACTS_DECAY_HOURS — параметры выбора «фактов» для объяснения.

NEWS_SENSITIVE_MODELS — список моделей, которым нельзя доверять высокие прогнозы без подтверждения новостей.

5. Telegram и платежи

TELEGRAM_BOT_TOKEN — токен бота.

TELEGRAM_ALERT_CHAT_ID — чат/канал для срочных уведомлений Position Guardian (можно использовать отдельный приватный канал).

TELEGRAM_PRIVATE_CHANNEL_ID — приватный канал, куда попадают подписчики после оплаты.

PRICE_STARS_MONTH, PRICE_STARS_YEAR — стоимость подписки в Telegram Stars (например, 500 и 5000). Цифровые товары продаются исключительно за Stars.

CRYPTO_PAYMENT_URL — ссылка на оплату в криптовалюте.

EXTERNAL_AFF_LINK_URL — ссылка на внешнюю партнёрскую биржу, добавляется в конец сообщений и подписей.
EXTERNAL_AFF_FOOTER_EN, EXTERNAL_AFF_FOOTER_RU — текст префикса для ссылки (по умолчанию «Recommended exchange:» и «Рекомендуемая биржа:»).


TELEGRAM_PRIVATE_CHANNEL_ID, TELEGRAM_ADMIN_CHAT_ID, TELEGRAM_OWNER_ID, TELEGRAM_ADMIN_IDS — каналы для подписки, админских сообщений, владельца и список ID админов.

MONTH_STARS, YEAR_STARS — стоимость подписки в Telegram Stars (например, 500 и 5000).

CRYPTO_PAYMENT_LINK — ссылка на оплату в криптовалюте.
CRYPTO_PAY_SECRET — секрет для проверки HMAC вебхука сервиса оплаты криптой (используется `/confirm` в `crypto_pay_api`).

AFF_UNIT_TO_USD — коэффициент перевода «младших» единиц платежа (например, Stars) в приблизительные USD для аффилиат‑метрик (по умолчанию 0 — отключено).

ENABLE_CRYPTO_PAY — включает кнопку «Оплатить криптой» в боте (по умолчанию 0 — кнопка скрыта, используйте платежи Telegram/Stars). Для реальной интеграции требуется CRYPTO_PAY_API_URL и корректная обработка вебхуков.
CRYPTO_PAY_SECRET — секрет для проверки HMAC вебхука сервиса `crypto_pay_api`.

TELEGRAM_PRIVATE_CHANNEL_ID, TELEGRAM_ADMIN_CHAT_ID, TELEGRAM_OWNER_ID, TELEGRAM_ADMIN_IDS — каналы для подписки, админских сообщений, владельца и список ID админов.
TELEGRAM_PRIVATE_CHANNEL_ID, TELEGRAM_ADMIN_CHAT_ID, TELEGRAM_OWNER_ID — каналы для подписки, админских сообщений и владельца.

PRICE_STARS_MONTH, PRICE_STARS_YEAR — стоимость месячной и годовой подписки в Telegram Stars (милли‑звёздах, например, 2500 и 25000 ≈ $25/$250).

CRYPTO_PAYMENT_URL — ссылка на оплату в криптовалюте.

Оплата: подписку можно оплатить звёздами через Telegram или криптовалютой; бот выдаёт адрес для перевода и активирует доступ после подтверждения.

Flowise endpoints (LLM)

- Базовый URL внутри docker-сети: FLOWISE_BASE_URL=http://flowise:3000
- Для каждого Chatflow возьмите его flowId и укажите полный endpoint вида:
  - FLOWISE_SENTIMENT_URL=http://flowise:3000/api/v1/prediction/<SENTIMENT_FLOW_ID>
  - FLOWISE_FORECAST_URL=http://flowise:3000/api/v1/prediction/<FORECAST_FLOW_ID>

Real‑time triggers

- TRIGGER_WATCH_NEWS — включает обработку новостных триггеров (по умолчанию 1).
- TRIGGER_NEWS_POLL_SEC — период фонового опроса новостей при включённой опции (по умолчанию 120 секунд).
- TRIGGER_NEWS_POLL_ENABLE — включает лёгкий внутренний опрос новостей в `trigger_agent` (по умолчанию 0). По умолчанию агент читает готовые новости из БД, а сбор новостей рекомендуется запускать отдельным пайплайном.
  - FLOWISE_EXPLAIN_URL=http://flowise:3000/api/v1/prediction/<EXPLAIN_FLOW_ID>
  - FLOWISE_DEBATE_URL=http://flowise:3000/api/v1/prediction/<DEBATE_FLOW_ID>
  - FLOWISE_SCENARIO_URL=http://flowise:3000/api/v1/prediction/<SCENARIO_FLOW_ID>
  - FLOWISE_VALIDATE_URL=http://flowise:3000/api/v1/prediction/<VALIDATE_FLOW_ID>
- Если endpoints не заданы, соответствующие LLM‑части будут пропущены с предупреждением, остальная логика отработает.

Личные сообщения вместо канала: если не хотите использовать канал/чат, оставьте TELEGRAM_CHAT_ID пустым и задайте список получателей в TELEGRAM_DM_USER_IDS (через запятую). Бот сможет писать им напрямую только если эти пользователи предварительно нажали /start у бота (Telegram не позволяет боту первым начать диалог).

Пользовательские инсайты: активные подписчики могут отправлять инсайты боту (меню «Инсайт»). Инсайт проходит оценку правдивости/актуальности (LLM при наличии FLOWISE_VALIDATE_URL, либо эвристика) и учитывается в новостных сигналах при формировании прогноза.


6. Наблюдаемость и логирование

PROM_PUSHGATEWAY_URL — URL Pushgateway Prometheus. Метрики (pipeline_step_seconds, pipeline_value, llm_calls, llm_failures) отправляются сюда.

7. Адаптивная оркестрация и стратегические данные

USE_MASTER_ORCHESTRATOR — включает интеллектуальный оркестратор (MasterOrchestratorAgent). При 1 основной цикл `scheduled_runner` строит план запуска агентов, при 0 используется прежняя логика `run_master_flow`.

STRATEGIC_DATA_KEYWORDS — список ключевых слов (через запятую) для агента стратегического управления данными. Используется при проактивном поиске источников.

PROJECT_TASKS_WEBHOOK — необязательный вебхук для отправки задач, созданных StrategicDataAgent, во внешнюю систему управления проектами (Jira/Linear/Notion). Если не задан, задачи логируются только в базе.

MODEL_ARTIFACT_ROOT — каталог, из которого подгружаются артефакты моделей (например, ONNX для динамических индикаторов). По умолчанию `artifacts` в корне проекта.
RED_TEAM_STRATEGY — стратегия, применяемая Красной Командой при бэктесте синтетических сценариев (формат `module:ClassName`, по умолчанию `pipeline.trading.backtest.strategies:MovingAverageCrossStrategy`).
RED_TEAM_BASE_PRICE — базовая цена (USD) для генерации синтетических рядов при стресс‑тестах (по умолчанию 30000).
SAFE_MODE_RISK_PER_TRADE — доля капитала для сделки в safe-mode (например, 0.003).
SAFE_MODE_LEVERAGE_CAP — максимальное плечо в safe-mode (по умолчанию 5).
ORCHESTRATOR_EVENT_POLL_SEC — период опроса очереди событий фонового слушателя (по умолчанию 10 секунд).

SENTRY_DSN — ключ для Sentry. При задании трейсбек ошибок отправляются в Sentry.

JSON_LOGS — если 1, loguru пишет логи в JSON формате. Укажите LOG_LEVEL при необходимости.

7. Флаги функций и ML

USE_KATS, USE_CPD_SIMPLE, CPD_Z_THRESHOLD, CPD_WINDOW — включают расширенные (Kats) или простые методы обнаружения режимов/аноналий.

FEATURES_PRO, FEATURES_KATS — включают расширенный набор признаков (например, volume profile, carry) и Kats‑фичи. Отключайте, если хотите ускорить вычисления.

ENABLE_STACKING — включает обучение линейного стэкинга по историческим прогнозам. Требует накопления ≥30 наблюдений.
ENSEMBLE_SIMILAR_BONUS — бонус к весам моделей (0..1) по результатам на «похожих окнах» (по умолчанию 0.2).
SIMILAR_TOPK_LIMIT — сколько «соседей» учитывать при динамическом взвешивании (по умолчанию 5).

ENABLE_CHALLENGER, CHALLENGER_MODE — активируют вторую ветку ансамбля (uniform/stacking) для A/B сравнения.

ENABLE_RETRAIN, RETRAIN_INTERVAL_H, RETRAIN_FEATURES_S3, RETRAIN_HORIZONS — расписание и параметры переобучения ML моделей. Сборка формирует единый бандл для всех указанных горизонтов (по умолчанию `1h,4h,24h`).
CONTEXT_TOKENS_BUDGET — ограничение на размер оперативной памяти в токенах (дефолт 32000).
CONTEXT_FETCH_TA / CONTEXT_FETCH_SMC — разрешить автоматический пересчёт продвинутого ТА и SMC при сборке контекста (`1`/`0`).
ARB_ANALYST_ENABLED, ARB_ANALYST_AB_PERCENT — включение цепочки рассуждений и процент трафика, попадающий в новый контур.
OPENAI_MODEL_ANALYST, OPENAI_MODEL_SELFCRIT — модели для CoT и самокритики (по умолчанию берутся из `OPENAI_MODEL_MASTER`).
ENABLE_SELF_CRITIQUE — включает SelfCritiqueAgent (по умолчанию 1).

USE_DARTS, USE_DARTS_NBEATS, NBEATS_EPOCHS, USE_NEURALPROPHET, NP_EPOCHS, USE_PROPHET, USE_DARTS_PROPHET — активируют альтернативные временные модели. Настройки эпох позволяют ускорить тренировку.

AUTO_TUNE_ARIMA, ARIMA_TUNING_TRIALS — запуск Optuna‑поиска гиперпараметров ARIMA.

MLFLOW_TRACKING_URI, ML_MODELS_S3_PREFIX, ML_MODEL_S3 — путь к трекеру MLflow и хранилищу моделей. Если задан, сохранённые модели и метаданные отправляются в реестр model_registry.

8. Trading и risk

PAPER_SYMBOL — торговая пара для симуляции бумаги (BTC/USDT).

EXCHANGE, EXCHANGE_TYPE, EXCHANGE_API_KEY, EXCHANGE_SECRET, EXCHANGE_PASSWORD — конфигурация ccxt для живой торговли.

EXEC_SYMBOL — символ для отправки ордеров.

EXECUTE_LIVE — флаг разрешения живой торговли.

EXEC_AMOUNT — базовый размер позиции.

EXEC_ENTRY — market или limit.

DRY_RUN — если 1, ордера не отправляются (бумажный режим).

Live Portfolio Manager
----------------------

- PORTFOLIO_LIVE — включить учёт live‑позиций через ccxt в `trade_recommend` (0/1).
- LIVE_PM_CACHE_SEC — TTL кэша ответов ccxt (по умолчанию 5 сек).
- LIVE_PM_USE_PAPER — вместо ccxt читать состояние из таблиц paper_* (для отладки).
- EXEC_DYNAMIC_SIZE — динамический размер позиции в `executor_live` на основе `EXEC_RISK_PER_TRADE`.
- EXEC_RISK_PER_TRADE — риск на сделку (доля equity), по умолчанию 0.01.

Lessons (петля обратной связи)
------------------------------

- OPENAI_EMBED_MODEL — модель эмбеддингов (по умолчанию `text-embedding-3-small`).
- Для сохранения векторных представлений уроков требуется миграция `040_agent_lessons_vector.sql` и расширение `vector` в PostgreSQL.

RAG (Knowledge Core)
--------------------

- ENABLE_RAG — включить поиск знаний (RAG) и подмешивание фактов в дебаты (1/0).
- Загрузка материалов: `pipeline.knowledge.loader.load_and_embed` — путь к txt/md.

Vault (секреты)
---------------

- VAULT_URL, VAULT_TOKEN, VAULT_PATH_PREFIX — включают чтение секретов через `infra/secrets.get_secret`.
- Используется для: OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, EXCHANGE_API_KEY/SECRET/PASSWORD.

Портфель и политики
-------------------

- PORTFOLIO_ENABLED — включает учёт портфеля и политики в `trade_recommend` (0/1).
- TR_SCALEIN_CONF_THRES — порог уверенности (0..1) для добора к существующей позиции (по умолчанию 0.68).
- TR_SCALEIN_FRACTION — доля базового размера при доборе (по умолчанию 0.5).

SMC и визуализация
------------------

- SMC_OHLC_CACHE_SEC — TTL кеша OHLCV в `SMC Analyst` (секунды).
- SMC_SAVE_ZONES — если 1, сохранять обнаруженные зоны в таблицу `smc_zones`.

Пост‑анализ сделок
-------------------

- POST_MORTEM_ENABLED — если 1, запускать PostMortemAgent после закрытия сделки и сохранять уроки в `agent_lessons`.

RISK_LOOP_INTERVAL, RISK_TRAIL_PCT, RISK_IMPROVE_PCT — параметры trailing‑stop.
RISK_LOOP_MODE — `trail` (по умолчанию) или `guardian` для сопровождения открытых сделок.
GUARDIAN_INTERVAL_MIN — интервал проверок guardian в минутах (по умолчанию 15).
GUARDIAN_PROB_THRESHOLD — минимальная вероятность успеха, при падении ниже которой отправляется алерт (например, 0.28).

9. Прочее

RUN_INTERVAL_SEC — интервал планировщика для scheduled_runner.py (3600 с по умолчанию).

USE_COORDINATOR — включает многоагентный DAG; в противном случае используется линейный скрипт predict_release.

USE_MASTER_AGENT — включает MasterAgent как центральную точку входа (возможен запуск отдельно: `python -m pipeline.agents.master`).

ENABLE_MCP — поднимает мини‑сервер MCP (pipeline.mcp.server), который предоставляет безопасные эндпоинты для LLM‑агентов (автозапуск из планировщика при ENABLE_MCP=1).

ENABLE_MULTI_DEBATE — включает «мульти‑агентные дебаты» (бычий/медвежий/квант) с итоговой агрегацией арбитром.

MCP_HOST, MCP_PORT, MCP_URL — адрес и порт MCP.

Заполняйте .env исходя из своих потребностей: выключайте тяжёлые источники (onchain, social, alt) для ускорения, включайте Flowise endpoints для генерации объяснений и дебатов, настраивайте лимиты LLM. Список не исчерпывающий — см. .env.example в репозитории для остальных переменных и значений по умолчанию.

Модели LLM и токены

- OPENAI_MODEL — дефолтная модель (рекомендуется `gpt-4o-mini` для быстрой обработки).
- OPENAI_MODEL_MASTER — модель для MasterAgent/арбитра (рекомендуется `gpt-4o` для сложных рассуждений).
- OPENAI_MODEL_VISION — мультимодальная модель для `chart_vision_agent`.
- OPENAI_MODEL_NEWS_FACTS — модель для news_facts (по умолчанию `gpt-4o-mini`).
- ENABLE_CHART_VISION_LLM — 1/0, включает вызов мультимодальной модели; при 0 агент использует эвристику.
- LLM_MAX_TOKENS — лимит на ответ (4096 по умолчанию — «пространство для мысли»).

Order Flow (тик‑стрим)

- ENABLE_ORDER_FLOW — включить сбор сделок (по умолчанию 0).
- ORDERFLOW_WINDOW_SEC — длина окна для сбора (по умолчанию 60 сек).
- ORDERFLOW_POLL_SEC — период REST‑опроса (по умолчанию 1 сек).
- ORDERFLOW_USE_WS — использовать WebSocket‑стрим через ccxt.pro (если установлен) вместо REST‑опроса (0/1).

Real‑Time режим (Trigger → Master)

- RT_TRIGGER_QUEUE — ключ Redis‑очереди для триггеров (по умолчанию `rt:triggers`).
- TRIGGER_SYMBOL — символ для мониторинга триггер‑агентом (по умолчанию берётся `PAPER_SYMBOL`).
- TRIGGER_POLL_SLEEP — пауза между циклами триггер‑агента (сек, по умолчанию 5).
- TRIGGER_VOL_SPIKE_FACTOR — порог срабатывания по объёму: (объём_минуты / EMA_24ч) ≥ factor (по умолчанию 3.0).
- TRIGGER_VOL_EMA_ALPHA — сглаживание EMA для базовой линии объёма (по умолчанию 0.03).
- TRIGGER_DELTA_BASE_MIN — минимальная |дельта_5м| в базовой валюте для триггера DELTA_SPIKE (по умолчанию 20.0).
- TRIGGER_COOLDOWN_SEC — антидребезг: таймаут между триггерами одного типа (по умолчанию 300 сек).
- TRIGGER_WATCH_NEWS — включить детекцию новостей (по умолчанию 1).
- TRIGGER_NEWS_IMPACT_MIN — минимальный impact_score для новостного триггера (по умолчанию 0.7).
- TRIGGER_NEWS_LOOKBACK_SEC — окно поиска свежих новостей (по умолчанию 120 сек).
- TRIGGER_ADAPTIVE — адаптация порогов по результатам paper PnL (по умолчанию 1).
- TRIGGER_ADAPT_EVERY_SEC — период пересчёта приоритетов триггеров (по умолчанию 300 сек).
- TRIGGER_L2_ENABLE — включить L2‑триггеры по стакану (по умолчанию 1).
- TRIGGER_L2_DEPTH_LEVELS — глубина уровней для суммирования (по умолчанию 50).
- TRIGGER_L2_NEAR_BPS — радиус близости к mid для «стены» в б.п. (по умолчанию 10 б.п.).
- TRIGGER_L2_WALL_MIN_BASE — минимальный размер «стены» в базовой валюте (по умолчанию 50.0 BTC).
- TRIGGER_L2_IMBALANCE_RATIO — порог дисбаланса суммарных bid/ask в топ‑K (по умолчанию 2.0).
- TRIGGER_NEWS_POLL_SEC — как часто триггер‑агент опрашивает новости (по умолчанию 120 сек).
- TRIGGER_NEWS_WINDOW_H — окно новостей при лёгком опросе (по умолчанию 1 час).

- RT_HORIZON_DEFAULT_MIN — дефолтный горизонт real‑time прогноза в минутах (по умолчанию 30).
- RT_HORIZON_VOL_SPIKE_MIN — горизонт для VOL_SPIKE (по умолчанию 30).
- RT_HORIZON_DELTA_SPIKE_MIN — горизонт для DELTA_SPIKE (по умолчанию 30).
- RT_HORIZON_NEWS_MIN — горизонт для NEWS (по умолчанию 60).
- RT_PRICES_LOOKBACK_SEC — глубина цен для real‑time анализа (по умолчанию 172800 = 48ч).
- RT_TIMEFRAME — таймфрейм для загрузки цен (по умолчанию `1m`).
- RT_NEWS_WINDOW_H — окно новостей для расчёта признаков (по умолчанию 6 часов).
- RT_VALID_FOR_MIN — время валидности карточки real‑time сигнала (по умолчанию 30 мин).

Telegram публикация

- TELEGRAM_RT_CHAT_ID — при задании RT‑сигналы отправляются в отдельный чат/канал, иначе используется общий `TELEGRAM_CHAT_ID`/DM.

Paper Trading (цикличность и исполнение)

- PAPER_EXEC_LOOKBACK_DAYS — за сколько дней подбирать новые рекомендации для открытия в исполнителе (по умолчанию 3).
- PAPER_RISK_PER_TRADE — риск на сделку в paper (доля от equity, по умолчанию 0.005).

TradeRecommend (тюнинг)

- TR_K_ATR — множитель ATR для SL (по умолчанию 1.5; может переопределяться автотюнером через agent_configurations).
- TR_RR_TARGET — целевой R:R (по умолчанию 1.6; может переопределяться автотюнером через agent_configurations).
- TR_TUNE_WINDOW_DAYS — окно для тюнера по paper (по умолчанию 30).

Feedback/Outcomes

- RT_OUTCOMES_HORIZONS — список RT горизонтов для сбора outcomes (по умолчанию `30m,60m`).
- RUN_PAPER_ADAPT — включить адаптацию trust‑весов по paper PnL в планировщике (по умолчанию 1).

Кэш y_true для ускорения динамического взвешивания

- YTRUE_CACHE_TTL_SEC — TTL кэша (в Redis) для фактических цен y_true, вычисляемых через CCXT при оценке похожих окон (по умолчанию 86400 сек).
