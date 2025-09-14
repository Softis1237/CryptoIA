## Архитектура

Высокоуровневая схема:

- orchestration/agent_flow.py — DAG агентов, сбор результатов, публикация, метрики
- data/* — источники (цены, новости, соцсети, ончейн, альт‑данные, стакан, фьючерсы, open interest/funding)
- features/* — генерация фич (тех.индикаторы, стакан/ончейн/соц‑агрегаты, контекст, supply/demand)
- models/* — модели (ETS/ARIMA + опционально Darts/NP/Prophet + ML‑бандлы)
- ensemble/* — ансамблирование
- regime/* — режим рынка (эвристика + ML‑placeholder)
- similarity/* — эмбеддинги в pgvector и поиск похожих окон
- scenarios/* — генератор сценариев (LLM/эвристика) с контекстом
- reasoning/* — объяснения и арбитраж (LLM/эвристика) с памятью/доверием
- trading/* — карточка сделки, верификатор, paper/live исполнение, оптимизатор, risk‑loop
- infra/* — БД/pgvector, S3, метрики, логгер, http‑ретраи, feature store (Redis)
- ops/* — Prometheus/Grafana/Windmill/Prom‑alerts
- migrations/* — SQL‑схемы

В модуле features_supply_demand агрегируются ключевые метрики баланса рынка:
дисбаланс стакана bid/ask, открытый интерес, ставка финансирования и доли
долгосрочных/краткосрочных держателей on-chain.

Поток релиза:

1) Ingest (prices/news/…)
2) FeaturesCalc (тех+контекст)
3) Regime/SimilarPast/Models→Ensemble
4) Scenarios/Chart
5) TradeRecommendation→Verifier
6) Validation Router + валидаторы (trade/regime/llm)
7) Debate/Explain, Risk Estimator
8) Publish (Telegram) + persist (DB) + metrics

S3 пути: `runs/{YYYY-MM-DD}/{slot}/{source}.parquet|png`.

Postgres: см. docs/DB_SCHEMA.md (перечень таблиц).
