# Стратегические агенты (SMC Analyst, Whale Watcher)

## БД‑интеграция

- Таблица `strategic_verdicts` (см. `migrations/033_strategic_verdicts.sql`):
  - Ключ: (`agent_name`, `symbol`, `ts`)
  - Поля: `verdict`, `confidence`, `meta`
- API в коде: `upsert_strategic_verdict(...)`, `fetch_latest_strategic_verdict(agent, symbol)`.

Дополнительно (опционально): выделить нормализованные таблицы для детальных полей `meta` и создать индексы по `created_at`/`symbol`.

## Планировщик

Рекомендуемое расписание:
- SMC Analyst: каждые 5 минут (HTF/LTF скан достаточно инерционен).
- Whale Watcher: предпочтительно «постоянно» (stream), либо 1 мин (минимум). Для постоянного режима:
  - подписка на потоки крупных сделок (CCXT/websocket), порог по размеру сделки;
  - подписка на on-chain события (Dune/webhook), порог по нет‑флоу.
  - при срабатывании — немедленный расчёт вердикта и upsert в БД.

Готовый скелет стрима:

```
PYTHONPATH=services/pipeline/src python -m pipeline.agents.whale_stream
```

ENV:
- `STREAM_SYMBOL` (по умолчанию `BTC/USDT`)
- `CCXT_EXCHANGE` (например, `binance`)
- `WHALE_TRADE_USD` (порог сделки в USD, по умолчанию 250000)
- `WHALE_POLL_SEC` (интервал фолбэк‑пулинга, по умолчанию 5)

Пример (одноразовый запуск из CLI):
```
PYTHONPATH=services/pipeline/src python -m pipeline.agents.smc_analyst '{"run_id":"smc_$(date +%s)","symbol":"BTC/USDT"}'
PYTHONPATH=services/pipeline/src python -m pipeline.agents.whale_watcher '{"run_id":"whale_$(date +%s)","symbol":"BTC/USDT"}'
```

Интеграция в Reactor:
- Реактор при каждом RT‑сигнале подтягивает последние вердикты SMC/Whale (по символу) и использует их в Confidence Score и в тексте алерта.

## Окно согласования с RT‑алертами

- Для повышения качества — учитывать только вердикты, чья `ts` попадает в окно `RT_ALERT_WINDOW_SEC` до текущего события (фильтрация на стороне Reactor).

## Наблюдаемость

- Писать метрики количества/доли бычьих/медвежьих вердиктов по агентам, сравнивать с результатами сделок.
- Еженедельный отчёт точности по каждой связке «режим рынка × агент».
