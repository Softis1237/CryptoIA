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
- ingest_news — CryptoPanic, NewsAPI, RSS (CoinDesk/Cointelegraph), Twitter, Reddit, StackOverflow; sentiment (HF/VADER+TextBlob); confidence

Соцсети:
- ingest_social — Twitter/Reddit, sentiment/темы/метрики; опционально эмбеддинги (pgvector)

Ончейн:
- ingest_onchain — Glassnode (active_addresses, netflow, mvrv_z, sopr, miners_balance, transfers_volume)

Альт‑данные:
- ingest_altdata — pytrends, CME OI (Quandl), опционный put/call (Coinglass), ликвидность/слippage, инциденты/регуляторные события (RSS)

ENV ключи см. docs/ENV.md
