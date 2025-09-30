# Live Trading Checklist

## Перед стартом

- ✅ Бумажная торговля: profit factor > 1.2, max drawdown < 10% за последние 60 дней.
- ✅ План отката: инструкция как отключить `ARB_ANALYST_ENABLED` и переключиться на legacy арбитра.
- ✅ Настроены ключи API биржи (ccxt) с минимальными правами (trade, без withdrawals).
- ✅ `EXECUTE_LIVE=1`, `PORTFOLIO_LIVE=1`, `EXEC_SYMBOL=BTC/USDT`.
- ✅ Настроены лимиты риска:
  ```env
  LIVE_MAX_POSITION=0.1   # BTC
  LIVE_MAX_DAILY_LOSS=200 # USDT
  ```
- ✅ Новые алерты в Grafana: «Live order fail», «Max loss exceeded».

## День запуска

1. Проверить соединение с биржей (`python -m pipeline.trading.exchange --ping`).
2. Сделать dry-run `predict_release.py` с `EXECUTE_LIVE=0` (sanity check).
3. Включить `EXECUTE_LIVE=1`; убедиться, что executor стартует без ошибок.
4. Контролировать первую сделку вручную:
   - ордер появился на бирже?
   - стоп-лосс/тейк-профит выставлены?
   - позиция отображается в Grafana?

## После запуска

- Мониторить `live_trades` и `live_positions` в БД.
- При инциденте: выключить `EXECUTE_LIVE`, переключиться на бумажный режим (`PORTFOLIO_LIVE=0`, `PAPER_TRADING_ENABLED=1`).
- Раз в неделю: выгрузить reasoning/critique и сверить с результатами.
- Постепенно увеличивать объём (не чаще раза в месяц). Старт: 1/10 целевого капитала.
