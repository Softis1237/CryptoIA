# Масштабное бэктестирование (Roadmap V5.0)

## Цель

Получить статистически значимые метрики (profit factor, Sharpe, max drawdown, win rate) на периоде 2022–2023 гг. с использованием нового аналитического контура (CoT + SelfCritique).

## 1. Подготовка данных

1. Скачайте исторический OHLCV (1h или 4h) за 2021-01-01..2023-12-31.
   ```bash
   mkdir -p data/backtest
   python scripts/download_binance.py --symbol BTCUSDT --interval 1h --start 2021-01-01 --end 2023-12-31 --out data/backtest/btc_1h.parquet
   ```
2. Убедитесь, что в датасете есть столбцы `timestamp`, `open`, `high`, `low`, `close`, `volume`.

## 2. Параметры окружения

- `ARB_ANALYST_ENABLED=1`, `ARB_LOG_TO_DB=0`, `ARB_STORE_S3=0` — чтобы бэктест не писал в БД.
- `OPENAI_MODEL_ANALYST`, `OPENAI_MODEL_SELFCRIT` — используйте локальный мок (см. `tests/agents/test_investment_arbiter.py`) или реальный ключ.
- `CONTEXT_FETCH_TA=1`, `CONTEXT_FETCH_SMC=1` — при наличии фич.

## 3. Запуск

```bash
bash ops/backtest_full.sh data/backtest/btc_1h.parquet artifacts/backtest
```

По умолчанию используется стратегия `MovingAverageCrossStrategy`. Для подключения полноценного конвейера реализуйте свою стратегию, которая на каждом баре вызывает `pipeline.orchestration.master.run_master_flow` с фиктивным run_id и получает карточку сделки.

## 3.1 Кросс-валидация и forecast_quality

- Перед масштабным бэктестом прогоните `python -m pipeline.ops.models_cv` (см. раздел AgentFlow) — проверяет модели на out-of-sample окнах.
- В Grafana откройте дашборд `forecast_quality` и зафиксируйте MAE/SMAPE/DA до и после изменения промптов.
- После бэктеста сравните историческую и свежую (последние 30 запусков) статистику — расхождения >20% сигнализируют о переобучении.

## 4. Анализ результатов

В `artifacts/backtest/summary.json` доступны:

- `profit_factor`
- `sharpe`
- `max_drawdown`
- `win_rate`
- перечень сделок

Сравните с baseline (moving average). Ожидаемое улучшение — хотя бы +15% к win rate и превышение profit factor > 1.5.

## 5. Документирование

Заполните таблицу:

| Дата запуска | Данные | Настройки | Profit Factor | Sharpe | Max DD | Win rate | Комментарии |
|--------------|--------|-----------|---------------|--------|--------|----------|-------------|

Добавьте графики equity и drawdown в отчет (можно воспользоваться Jupyter).
