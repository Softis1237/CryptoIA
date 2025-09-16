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
