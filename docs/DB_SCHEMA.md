## Основные таблицы

- prices_agg — агрегированные цены (MVP)
- features_snapshot — ссылка на последнюю выгрузку фич (S3)
- regimes — последнее состояние режима
- similar_windows — соседи по эмбеддингу (pgvector)
- predictions — итоговые предсказания (4h/12h) с per_model_json
- ensemble_weights — веса ансамбля
- explanations — итоговое объяснение (markdown + risk flags)
- scenarios — список сценариев и ссылка на график
- trades_suggestions — карточка сделки
- news_signals — сигналы новостей (confidence/impact)
- social_signals — сигналы соцсетей
- onchain_signals — ончейн метрики
- futures_metrics — фьючерсные метрики
- features_index — эмбеддинги окон (pgvector)
- agents_predictions — JSON‑результаты агентов
- agents_metrics — числовые метрики агентов (Prometheus‑friendly)
- validation_reports — итог многоуровневой валидации
- backtest_results — результаты бэктестов валидатора
- model_registry — версии моделей с параметрами/метриками/путями
- users_feedback — обратная связь пользователей
- users — предпочтения пользователей/таймзона/активы
- redeem_codes — коды подписки на N месяцев

См. папку `migrations/` для SQL определения и индексов.
