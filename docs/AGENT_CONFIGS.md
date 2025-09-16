# Конфигурация агентов через agent_configurations

Таблица `agent_configurations` хранит активные параметры/промпты агентов. Ниже — референсные конфиги и примеры вставки через Python REPL.

## AlertPriority

Назначение: на основе композиции триггеров за короткое окно (Redis) присвоить приоритет алерту.

Параметры:
- `weights`: веса отдельных триггеров (по умолчанию заданы в коде, можно переопределить).
- `thresholds`: пороги score для low/medium/high/critical.
- `critical_combos`: список комбинаций триггеров, необходимых для статуса `critical`.

Базовый пример:
```
{
  "weights": {
    "VOL_SPIKE": 1.0,
    "DELTA_SPIKE": 1.0,
    "ATR_SPIKE": 1.2,
    "MOMENTUM": 1.3,
    "NEWS": 1.4,
    "L2_WALL": 0.8,
    "L2_IMBALANCE": 1.0,
    "PATTERN_BULL_ENGULF": 1.2,
    "PATTERN_BEAR_ENGULF": 1.2,
    "PATTERN_HAMMER": 1.0,
    "PATTERN_SHOOTING_STAR": 1.0,
    "DERIV_OI_JUMP": 1.2,
    "DERIV_FUNDING": 1.1
  },
  "thresholds": {"low": 0.0, "medium": 1.0, "high": 2.0, "critical": 3.0},
  "critical_combos": [["MOMENTUM", "VOL_SPIKE"], ["MOMENTUM", "PATTERN_BULL_ENGULF"], ["MOMENTUM", "PATTERN_BEAR_ENGULF"]]
}
```

Пресеты по режиму рынка:
- Тренд: увеличьте веса `MOMENTUM`, `ATR_SPIKE`, снизьте `L2_WALL`.
- Флэт/диапазон: повысьте `L2_IMBALANCE`, `PATTERN_*`, уменьшите `MOMENTUM`.

## ConfidenceAggregator

Назначение: агрегировать итоговую уверенность сигнала из базовой confidence и факторов контекста.

Параметры (веса факторов):
```
{
  "w_trigger": 0.08,
  "w_pattern": 0.10,
  "w_deriv": 0.08,
  "w_ta": 0.08,
  "w_regime": 0.05,
  "w_news": 0.05,
  "w_alert": 0.05,
  "w_smc": 0.10,
  "w_whale": 0.08,
  "w_alpha": 0.10
}
```

Рекомендации:
- Повышайте `w_smc` и/или `w_whale`, если стратегические агенты показывают высокую точность в недавнем периоде.
- Снижайте `w_news` в периоды низкой новостной волатильности.

## TriggerAgent

Назначение: мониторинг order flow/цены/деривативов и публикация триггеров в реальном времени.

Основные параметры (ключи в `parameters`):

| Ключ | Описание | Значение по умолчанию |
|------|----------|-----------------------|
| `symbol` | Базовый торговый инструментарий (`BTC/USDT`) | значение `PAPER_SYMBOL` или `BTC/USDT` |
| `orderflow_window_sec` | Окно агрегации order flow в секундах | 60 |
| `poll_sleep` | Пауза между циклами (сек) | 5.0 |
| `vol_spike_factor` | Множитель для триггера VOL_SPIKE | 3.0 |
| `delta_abs_min` | Порог суммарного дельта-объёма | 20.0 |
| `cooldown_sec` | Кулдаун между однотипными триггерами | 300 |
| `watch_news` | Включить реакцию на новости | true |
| `news_impact_min` | Минимальный impact score для NEWS | 0.7 |
| `adaptive_learning` | Адаптация приоритетов по PnL | true |
| `ccxt_provider` | Основной провайдер ccxt | `CCXT_PROVIDER` / `binance` |
| `fallback_providers` | Доп. провайдеры ccxt, список строк | [] |
| `orderflow_provider` | Провайдер для order flow | `FUTURES_PROVIDER` |
| `l2_provider` | Провайдер L2 стакана | `orderflow_provider` |
| `l2_enable` | Включить L2 триггеры | true |
| `l2_depth_levels` | Количество уровней стакана | 50 |
| `l2_near_bps` | Радиус поиска стенок (bps) | 10 |
| `l2_wall_min_base` | Минимальный размер стенки (base asset) | 50 |
| `l2_imbalance_ratio` | Порог дисбаланса | 2.0 |
| `news_poll_enable` | Фоновый опрос новостей | false |
| `news_poll_sec` | Период фонового опроса | 120.0 |
| `news_poll_window_h` | Окно фонового опроса (ч) | 1 |
| `atr_factor` | Множитель ATR для ATR_SPIKE | 3.0 |
| `momentum_5m_pct` | Порог изменения цены за 5 минут | 0.005 |
| `patterns_enable` | Включить свечные паттерны | true |
| `deriv_enable` | Включить деривативные аномалии | true |
| `deriv_oi_jump_pct` | Порог изменения OI | 0.05 |
| `deriv_funding_abs` | Порог абсолютного funding | 0.01 |
| `vol_ema_alpha` | alpha для EMA объёма | 0.03 |
| `news_lookback_sec` | Окно поиска новостей в БД | 120 |
| `adapt_every_sec` | Частота перерасчёта приоритетов | 300 |
| `ccxt_timeout_ms` | Таймаут запросов ccxt | 20000 |
| `ccxt_retries` | Количество повторов ccxt | 3 |
| `ccxt_retry_backoff_sec` | Базовый backoff (сек) | 1.0 |

Все параметры имеют fallback на переменные окружения и дальше — на значение по умолчанию.

## RealtimeMaster

Назначение: основной реактор real-time сигналов (`master_reactor.py`).

Параметры:

- `price_providers`: список ccxt-провайдеров, которые будут опрашиваться по очереди.
- `price_timeout_ms`: таймаут ccxt в миллисекундах (по умолчанию `CCXT_TIMEOUT_MS` или 20000).
- `price_retries`: количество повторов при ошибке провайдера (по умолчанию `CCXT_RETRIES` или 3).
- `price_backoff_sec`: экспоненциальный backoff между попытками (по умолчанию `CCXT_RETRY_BACKOFF_SEC` или 1.0).

Пример конфигурации:

```
{
  "price_providers": ["binance", "okx"],
  "price_timeout_ms": 20000,
  "price_retries": 3,
  "price_backoff_sec": 1.0
}
```

## Пример вставки/обновления активной версии (Python)

```
PYTHONPATH=services/pipeline/src python - <<'PY'
from pipeline.infra.db import insert_agent_config
alert_params = {"weights": {"MOMENTUM":1.3,"VOL_SPIKE":1.0}, "thresholds": {"low":0.0,"medium":1.0,"high":2.0,"critical":3.0}, "critical_combos": [["MOMENTUM","VOL_SPIKE"]]}
ver = insert_agent_config("AlertPriority", None, alert_params, make_active=True)
print("AlertPriority version:", ver)
conf_params = {"w_trigger":0.08,"w_pattern":0.10,"w_deriv":0.08,"w_ta":0.08,"w_regime":0.05,"w_news":0.05,"w_alert":0.05,"w_smc":0.10,"w_whale":0.08}
ver2 = insert_agent_config("ConfidenceAggregator", None, conf_params, make_active=True)
print("ConfidenceAggregator version:", ver2)
PY
```

Или одной командой:

```
make seed-configs
```

## ENV

- `RT_ALERT_WINDOW_SEC` — окно агрегации триггеров в секундах (по умолчанию 180).
