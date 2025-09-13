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

ENABLE_NEWS_TWITTER/REDDIT/STACKOVERFLOW — включение отдельных источников новостей.

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

TELEGRAM_PRIVATE_CHANNEL_ID — приватный канал, куда попадают подписчики после оплаты.

PRICE_STARS_MONTH, PRICE_STARS_YEAR — стоимость подписки в Telegram Stars (например, 500 и 5000). Цифровые товары продаются исключительно за Stars.

CRYPTO_PAYMENT_URL — ссылка на оплату в криптовалюте.


TELEGRAM_PRIVATE_CHANNEL_ID, TELEGRAM_ADMIN_CHAT_ID, TELEGRAM_OWNER_ID, TELEGRAM_ADMIN_IDS — каналы для подписки, админских сообщений, владельца и список ID админов.

MONTH_STARS, YEAR_STARS — стоимость подписки в Telegram Stars (например, 500 и 5000).

CRYPTO_PAYMENT_LINK — ссылка на оплату в криптовалюте.
CRYPTO_WEBHOOK_SECRET — секрет для проверки HMAC вебхука крипто‑сервиса.

AFF_UNIT_TO_USD — коэффициент перевода «младших» единиц платежа (например, Stars) в приблизительные USD для аффилиат‑метрик (по умолчанию 0 — отключено).

ENABLE_CRYPTO_PAY — включает кнопку «Оплатить криптой» в боте (по умолчанию 0 — кнопка скрыта, используйте платежи Telegram/Stars). Для реальной интеграции требуется CRYPTO_PAY_API_URL и корректная обработка вебхуков.
CRYPTO_WEBHOOK_SECRET — секрет для проверки HMAC вебхука крипто‑сервиса.

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
  - FLOWISE_EXPLAIN_URL=http://flowise:3000/api/v1/prediction/<EXPLAIN_FLOW_ID>
  - FLOWISE_DEBATE_URL=http://flowise:3000/api/v1/prediction/<DEBATE_FLOW_ID>
  - FLOWISE_SCENARIO_URL=http://flowise:3000/api/v1/prediction/<SCENARIO_FLOW_ID>
  - FLOWISE_VALIDATE_URL=http://flowise:3000/api/v1/prediction/<VALIDATE_FLOW_ID>
- Если endpoints не заданы, соответствующие LLM‑части будут пропущены с предупреждением, остальная логика отработает.

Личные сообщения вместо канала: если не хотите использовать канал/чат, оставьте TELEGRAM_CHAT_ID пустым и задайте список получателей в TELEGRAM_DM_USER_IDS (через запятую). Бот сможет писать им напрямую только если эти пользователи предварительно нажали /start у бота (Telegram не позволяет боту первым начать диалог).

Пользовательские инсайты: активные подписчики могут отправлять инсайты боту (меню «Инсайт»). Инсайт проходит оценку правдивости/актуальности (LLM при наличии FLOWISE_VALIDATE_URL, либо эвристика) и учитывается в новостных сигналах при формировании прогноза.


6. Наблюдаемость и логирование

PROM_PUSHGATEWAY_URL — URL Pushgateway Prometheus. Метрики (pipeline_step_seconds, pipeline_value, llm_calls, llm_failures) отправляются сюда.

SENTRY_DSN — ключ для Sentry. При задании трейсбек ошибок отправляются в Sentry.

JSON_LOGS — если 1, loguru пишет логи в JSON формате. Укажите LOG_LEVEL при необходимости.

7. Флаги функций и ML

USE_KATS, USE_CPD_SIMPLE, CPD_Z_THRESHOLD, CPD_WINDOW — включают расширенные (Kats) или простые методы обнаружения режимов/аноналий.

FEATURES_PRO, FEATURES_KATS — включают расширенный набор признаков (например, volume profile, carry) и Kats‑фичи. Отключайте, если хотите ускорить вычисления.

ENABLE_STACKING — включает обучение линейного стэкинга по историческим прогнозам. Требует накопления ≥30 наблюдений.
ENSEMBLE_SIMILAR_BONUS — бонус к весам моделей (0..1) по результатам на «похожих окнах» (по умолчанию 0.2).
SIMILAR_TOPK_LIMIT — сколько «соседей» учитывать при динамическом взвешивании (по умолчанию 5).

ENABLE_CHALLENGER, CHALLENGER_MODE — активируют вторую ветку ансамбля (uniform/stacking) для A/B сравнения.

ENABLE_RETRAIN, RETRAIN_INTERVAL_H, RETRAIN_FEATURES_S3, RETRAIN_HORIZONS — расписание и параметры переобучения ML моделей.

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

DRY_RUN — если 1, ордера не отправляются (бумажный режим).

RISK_LOOP_INTERVAL, RISK_TRAIL_PCT, RISK_IMPROVE_PCT — параметры trailing‑stop.

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
- OPENAI_MODEL_NEWS_FACTS — модель для news_facts (по умолчанию `gpt-4o-mini`).
- LLM_MAX_TOKENS — лимит на ответ (4096 по умолчанию — «пространство для мысли»).

Order Flow (тик‑стрим)

- ENABLE_ORDER_FLOW — включить сбор сделок (по умолчанию 0).
- ORDERFLOW_WINDOW_SEC — длина окна для сбора (по умолчанию 60 сек).
- ORDERFLOW_POLL_SEC — период REST‑опроса (по умолчанию 1 сек).
- ORDERFLOW_USE_WS — использовать WebSocket‑стрим через ccxt.pro (если установлен) вместо REST‑опроса (0/1).

Кэш y_true для ускорения динамического взвешивания

- YTRUE_CACHE_TTL_SEC — TTL кэша (в Redis) для фактических цен y_true, вычисляемых через CCXT при оценке похожих окон (по умолчанию 86400 сек).
