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
- knowledge_docs — база знаний (чанки документов + эмбеддинги pgvector)
- agents_predictions — JSON‑результаты агентов
- agents_metrics — числовые метрики агентов (Prometheus‑friendly)
- strategic_verdicts — последние вердикты стратегических агентов (SMC/Whale): (agent_name, symbol, ts, verdict, confidence, meta)
- smc_verdict_details / whale_verdict_details — нормализованные метрики стратегов
- elite_leaderboard_trades — сделки топ‑трейдеров (source/trader_id/symbol/side/ts/pnl)
- alpha_snapshots — контекстные снимки рынка вокруг сделок элит‑трейдеров
- alpha_strategies — формализованные стратегии Alpha Hunter
- validation_reports — итог многоуровневой валидации
- backtest_results — результаты бэктестов валидатора
- model_registry — версии моделей с параметрами/метриками/путями
- users_feedback — обратная связь пользователей
- users — предпочтения пользователей/таймзона/активы
- affiliates — партнёры (user_id, code, percent, balance)
- referrals — рефералы/начисления по первым покупкам (partner_user_id, referred_user_id, charge_id, amount, commission)
- users.referrer_code / users.referrer_name — код и имя партнёра, от которого пришёл пользователь
- affiliate_requests — заявки на партнёрку (user_id, username, note, status, created_at, processed_at)
- redeem_codes — коды подписки на N месяцев
- run_summaries — краткие итоги запусков (память агента MasterAgent)
 - technical_patterns — база знаний теханализа (канонические определения паттернов)
 - agent_configurations — версии промптов/параметров агентов
 - pattern_discovery_metrics — метрики валидации найденных паттернов
 - agent_lessons — сжатая память (уроки) на основе run_summaries

См. папку `migrations/` для SQL определения и индексов.

### Таблица: technical_patterns

### Таблица: technical_patterns

Назначение: централизованная «база знаний» определений свечных/графических паттернов, которую используют агенты и фичебилдер.

Схема (см. `migrations/025_technical_patterns.sql`):
- `pattern_id SERIAL PRIMARY KEY`
- `name TEXT UNIQUE NOT NULL` — каноническое имя (`hammer`, `double_top`, ...)
- `category TEXT NOT NULL` — `candle` или `chart`
- `timeframe TEXT NOT NULL DEFAULT '1m'`
- `definition_json JSONB NOT NULL` — формализованные правила (lookback/пороги/структура)
- `description TEXT` — текстовое описание
- `source TEXT` — источник/тип (`seed`, `discover`, `manual`)
- `confidence_default REAL NOT NULL DEFAULT 0.6`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`

Как добавить новый паттерн (два пути):
- SQL: `INSERT INTO technical_patterns (name, category, timeframe, definition_json, description, source, confidence_default) VALUES (...);`
- Через агента PatternDiscoveryAgent (см. Phase 2): он вставляет записи автоматически при достаточной статистической значимости.

Использование в фичах:
- Модуль `features_patterns.py` загружает определения (для будущих правил) и рассчитывает флаги `pat_*`.
- Текущая версия включает быстрые эвристики для свечных и граф‑паттернов; фичи публикуются как числовые столбцы `pat_*` в `features.parquet`.

Гайд по правилам/эвристикам:
- Свечные: соотношения теней/тела, последовательности (engulfing, morning star и др.).
- Граф‑паттерны: искать локальные экстремумы в последнем окне, проверить: (1) «похожесть» вершин/впадин, (2) минимальные интервалы между ними, (3) подтверждение пробоя «линии шеи/шейной линии» (neckline) последней ценой.
  - Пример double top: две вершины схожей высоты, между ними локальный минимум; подтверждение — последний close ниже «шеи».
  - Пример head and shoulders: левое плечо — голова — правое плечо; подтверждение — последний close ниже линии, соединяющей два минимума между плечами/головой.
### Таблица: agent_configurations

Назначение: хранение версий промптов и параметров для агентов (пример: ChartReasoningAgent, PatternDiscoveryAgent).

Схема (см. `migrations/026_agent_configurations.sql`):
- `agent_name TEXT`, `version INT`, `system_prompt TEXT`, `parameters_json JSONB`, `is_active BOOLEAN`, `created_at TIMESTAMPTZ`.
- Уникальность по (`agent_name`, `version`). Активная конфигурация выбирается по `is_active` и максимальной `version`.

Использование:
- `fetch_agent_config(agent_name)` — выбрать активную запись и переопределить системный промпт/параметры (модель, температура и т.п.).

### Таблица: pattern_discovery_metrics

Назначение: хранение статистической валидации кандидатов паттернов, найденных Discovery‑агентом.

Схема (см. `migrations/027_pattern_metrics.sql`):
- ключевые поля: `symbol`, `timeframe`, `window_hours`, `move_threshold`, `sample_count`, `pattern_name`, `expected_direction`,
  `match_count`, `success_count`, `success_rate`, `p_value`, `definition_json`, `summary_json`.

Методика:
- Для каждого кандидата определяется набор `selection_rules` (ссылки на агрегаты из предокна: `count_pat_*`, `rsi_mean`, `bb_width_mean`, ...).
- Выбираем матчи по правилам среди событий, считаем долю успехов (знак движения совпадает с ожидаемым) и p‑value биномиального теста против p0=0.5.
- Порог (пример): `match_count ≥ 8`, `success_rate ≥ 0.65`, `p_value ≤ 0.05` — можно авто‑добавлять в `technical_patterns`.

### Таблица: agent_lessons

Назначение: хранение кратких уроков/выжимок для MasterAgent, полученных из последних `run_summaries`.

Схема (см. `migrations/030_agent_lessons.sql`, `037_agent_lessons_structured.sql`):
- `created_at TIMESTAMPTZ`, `scope TEXT` (по умолчанию `global`), `lesson_text TEXT`, `meta JSONB DEFAULT '{}'::jsonb`.
- Частичный уникальный индекс `uq_agent_lessons_scope_hash` предотвращает дублирование карточек с одинаковым `meta->>'hash'` внутри одного `scope`.

Получение и наполнение:
- MCP инструмент `compress_memory(n?, scope?)` вызывает MemoryCompressor, который агрегирует последние N записей `run_summaries` в 3–7 уроков и сохраняет их.
- Hash урока сохраняется в `meta->>'hash'`, что позволяет фильтровать дубликаты на уровне БД.
- MCP инструмент `get_lessons(n?)` возвращает последние уроки.
