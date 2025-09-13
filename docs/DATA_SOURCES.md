## Источники данных

S3/MinIO: все партиции сохраняются в `runs/{YYYY-MM-DD}/{slot}/{source}.parquet|png`.

Цены:
- ingest_prices — минутные свечи (ccxt)
- ingest_prices_lowtf — 1s/5s, fallback агрегация из trades

Стакан:
- ingest_orderbook — snapshot топ‑N уровней + метрики (imbalance, объёмы)

Фьючерсы:
- ingest_futures — funding, mark/index, open interest

Новости:
- ingest_news — CryptoPanic, NewsAPI, RSS (fallback без ключей), Twitter, Reddit, StackOverflow.
  - RSS реализован на aiohttp+feedparser, есть дедупликация по URL и базовая эвристика тональности (bull/up/pump → positive; bear/down/hack/exploit → negative; иначе neutral). При `ENABLE_NEWS_SENTIMENT=1` используется LLM (Flowise endpoint `FLOWISE_SENTIMENT_URL`).
  - Источники: компактный дефолт (CoinDesk, Cointelegraph, Bitcoin Magazine, Blockworks, etc) + опциональный файл `pipeline/data/rss_sources_full.txt` (или `RSS_SOURCES_FILE`) с расширенным списком (вкл. русскоязычные медиа).
  - Настройки: `RSS_NEWS_SOURCES` для добавления/замены списком, `RSS_TIMEOUT_SEC`, `RSS_CONCURRENCY`.

Соцсети:
- ingest_social — Twitter/Reddit, sentiment/темы/метрики; опционально эмбеддинги (pgvector)

Ончейн:
- ingest_onchain — Dune Analytics (active_addresses, netflow, mvrv_z, sopr, miners_balance, transfers_volume) через публичные SQL‑запросы. Укажите `DUNE_API_KEY` и ID запросов в .env (см. DUNE_QUERY_*). Рекомендуемый формат результата каждого запроса: столбцы `ts` (timestamp/int, сек) и `value` (число). Если ключ не задан или `ENABLE_ONCHAIN=0`, сбор пропускается, и в сообщениях не показывается «0 сигналов».

Альт‑данные:
- ingest_altdata — pytrends, CME OI (Quandl), опционный put/call (Coinglass), ликвидность/слippage, инциденты/регуляторные события (RSS)

ENV ключи см. docs/ENV.md
