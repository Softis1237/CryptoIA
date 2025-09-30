# Бумажная торговля (Paper Trading)

## Цель
Запустить полный конвейер в реальном времени без финансового риска, используя бумажный портфель, и собирать метрики P&L.

## 1. Предварительные требования

- Применены миграции до `044_arbiter_reasoning.sql`.
- Заполнены `.env` параметры для Postgres/Redis/S3.
- `PORTFOLIO_ENABLED=1`, `PORTFOLIO_MODE=paper`, `PAPER_TRADING_ENABLED=1`.
- Настроены alert панели `trading_results.json` в Grafana.

## 2. Конфигурация

```env
ARB_ANALYST_ENABLED=1
ARB_LOG_TO_DB=1
ARB_STORE_S3=1
PAPER_TRADING_SAVE_TRADES=1
PAPER_SYMBOL=BTC/USDT
TRADING_TIMEFRAME=4h
```

## 3. Планировщик

Запуск `predict_release.py` каждые 4 часа (пример systemd таймера):

```ini
OnCalendar=00/4:00
ExecStart=/usr/bin/python -m pipeline.orchestration.predict_release --slot=paper
```

## 4. Мониторинг

- Grafana: панель `trading_results.json`, метрики `paper_pnl`, `paper_equity`.
- Prometheus алерты: «P&L < -5%» и «Нет сделок > 48ч».
- Проверяйте таблицы `paper_positions`, `paper_trades`, `arbiter_reasoning`.

## 5. Регулярные действия

| Частота | Действие |
|---------|----------|
| Ежедневно | Смотрите уведомления Telegram, проверяйте P&L. |
| Каждую неделю | Делайте выборку reasoning/critique и проводите ретроспективу качества. |
| Ежемесячно | Запускайте backtest на последних данных для сверки. |

## 6. Критерии готовности к живой торговле

- 1–2 месяца стабильного P&L (profit factor > 1.2, max drawdown < 10%).
- Нет критических сбоев в логах, телеметрии, reasoning.
- Команда понимает, как откатывать промпты и как реагировать на алерты.

## 7. Отладка

- Используйте `services/pipeline/ops/export_reasoning.py --limit 10` для анализа спорных сделок.
- Проверяйте `ops/trading/paper_trading.py` — в логах должны быть записи о каждой сделке.

После успешного прохождения бумажной торговли переходите к `docs/LIVE_TRADING_CHECKLIST.md`.
