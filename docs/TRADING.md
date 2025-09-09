## Торговые компоненты

Paper trading:
- executor_once / risk_loop / settler_loop / отчёты администратору
- Модули: pipeline.trading.paper_trading

Live (ccxt):
- exchange.py — обёртка над CCXT
- engine.py — брекет‑сделка (entry+SL/TP), DRY_RUN=1 по умолчанию
- risk_loop_live.py — каркас трейлинга
- ENV: EXCHANGE, EXCHANGE_TYPE, EXCHANGE_API_KEY/SECRET, EXEC_SYMBOL, DRY_RUN

Пример (dry‑run):

```
python -m pipeline.trading.engine BTC/USDT buy 0.01 --entry market --sl 60000 --tp 66000 --leverage 5
```

Оптимизация позиций:
- optimizer.py — Kelly/Equal‑risk аллокация по нескольким сигналам
