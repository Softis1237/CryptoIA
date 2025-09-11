ENV_FULL.md — подробная сводка переменных окружения

Этот файл агрегирует все переменные окружения, используемые в CryptoIA. В большинстве случаев необязательные компоненты отключены по умолчанию, и вы можете включить их, установив значение в 1 или присвоив API‑ключ.

1. Основная инфраструктура

POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT  — параметры подключения к PostgreSQL. Используются всеми модулями для сохранения результатов, сигналов, прогнозов и торговых позиций.

REDIS_HOST, REDIS_PORT — адрес кеша Redis для промежуточных данных, кэша LLM и блокировки операций.

S3_ENDPOINT_URL, S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET — настройки MinIO/S3. Сюда записываются parquet‑файлы с ценами и признаками, графики и модели.

TIMEZONE — таймзона слотов. По умолчанию Asia/Jerusalem. Используется для округления времени при публикации и расписания задач.

2. Источники данных и toggles

CRYPTOPANIC_TOKEN, NEWSAPI_KEY — токены для агрегаторов новостей CryptoPanic и NewsAPI.

ENABLE_ORDERBOOK (0/1) — загрузка стакана (orderbook). Требует Binance/Bybit/etc через ccxt.

ENABLE_ONCHAIN, GLASSNODE_API_KEY, CRYPTOQUANT_API_KEY — включение ончейн‑метрик и соответствующие ключи.

ENABLE_FUTURES, FUTURES_PROVIDER — сбор funding rate, mark price и open interest (default binance).

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

LLM_MAX_TOKENS — максимальное количество токенов в ответе (400 по умолчанию).

ENABLE_LLM_CACHE, LLM_CACHE_TTL_SEC — включение и время жизни кэша для LLM ответов (по умолчанию 600 с). Использует Redis.

LLM_CALLS_BUDGET — лимит вызовов LLM за процесс (8 по умолчанию).

FLOWISE_BASE_URL — базовый URL сервиса Flowise. Отдельные endpoints:

FLOWISE_EXPLAIN_URL, FLOWISE_DEBATE_URL, FLOWISE_SCENARIO_URL — конечные точки для генерации объяснения, дебатов и сценариев.

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

TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID — токен бота и ID канала/чата для публикаций.

TELEGRAM_PRIVATE_CHANNEL_ID, TELEGRAM_ADMIN_CHAT_ID, TELEGRAM_OWNER_ID — каналы для подписки, админских сообщений и владельца.

MONTH_STARS, YEAR_STARS — стоимость подписки в Telegram Stars (например, 500 и 5000).

CRYPTO_PAYMENT_LINK — ссылка на оплату в криптовалюте.

6. Наблюдаемость и логирование

PROM_PUSHGATEWAY_URL — URL Pushgateway Prometheus. Метрики (pipeline_step_seconds, pipeline_value, llm_calls, llm_failures) отправляются сюда.

SENTRY_DSN — ключ для Sentry. При задании трейсбек ошибок отправляются в Sentry.

JSON_LOGS — если 1, loguru пишет логи в JSON формате. Укажите LOG_LEVEL при необходимости.

7. Флаги функций и ML

USE_KATS, USE_CPD_SIMPLE, CPD_Z_THRESHOLD, CPD_WINDOW — включают расширенные (Kats) или простые методы обнаружения режимов/аноналий.

FEATURES_PRO, FEATURES_KATS — включают расширенный набор признаков (например, volume profile, carry) и Kats‑фичи. Отключайте, если хотите ускорить вычисления.

ENABLE_STACKING — включает обучение линейного стэкинга по историческим прогнозам. Требует накопления ≥30 наблюдений.

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

ENABLE_MCP — поднимает мини‑сервер MCP (pipeline.mcp.server), который предоставляет безопасные эндпоинты для LLM‑агентов.

MCP_HOST, MCP_PORT, MCP_URL — адрес и порт MCP.

Заполняйте .env исходя из своих потребностей: выключайте тяжёлые источники (onchain, social, alt) для ускорения, включайте Flowise endpoints для генерации объяснений и дебатов, настраивайте лимиты LLM. Список не исчерпывающий — см. .env.example в репозитории для остальных переменных и значений по умолчанию.
