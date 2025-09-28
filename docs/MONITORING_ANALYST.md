# Monitoring — Analyst Council

## Метрики Prometheus

| Metric | Type | Description |
|--------|------|-------------|
| `context_builder_tokens` | gauge | Оценка размера оперативной памяти (токены) по символу/слоту. |
| `context_builder_latency_ms` | gauge | Время сборки контекста. |
| `investment-analyst_probability_pct` | gauge | Вероятность, которую вернул CoT аналитик, с лейблом сценария. |
| `self-critique_probability_delta` | gauge | Корректировка вероятности оппонентом. |

## Пример панелей Grafana

1. **Context Size** – `avg(context_builder_tokens) by (symbol)`, отображение в виде столбцов.
2. **Analyst Probability** – `avg_over_time(investment-analyst_probability_pct[12h])` с разбивкой по сценариям.
3. **Critique Delta** – `sum_over_time(self-critique_probability_delta[6h])` для контроля излишней самокритики.
4. **Latency Heatmap** – `histogram_quantile(0.9, rate(context_builder_latency_ms_bucket[5m]))` (если подключить histogram).

## Алерты

- **Missing context**: `context_builder_tokens == 0` в течение 3 запусков подряд.
- **Critique Reject**: доля `self-critique_probability_delta < -10` более 30 % за час.

## Экспорт в Slack/Telegram

Используйте существующий `ops/prometheus/alerts.yaml`, добавив правила на новые метрики. Сигналы направляйте в канал поддержки трейдеров.
