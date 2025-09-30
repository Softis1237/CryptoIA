# Observability Guide

## Prometheus metrics
- LLM: `llm_failures`, `llm_validation` (type labels `analyst/critique`)
- Context: `context_builder_tokens`, `context_builder_latency_ms`
- Forecast: `forecast_quality_mae`, `forecast_quality_smape`, `forecast_quality_da`
- Agent performance: `agent_performance_weight`

## Alert rules (ops/prometheus/alerts.yml)
```yaml
 groups:
   - name: llm
     rules:
       - alert: LLMFailuresBurst
         expr: increase(llm_failures[15m]) > 3
         for: 5m
         labels:
           severity: critical
         annotations:
           summary: "LLM failures spike"
           description: "Check OpenAI/Flowise availability"
       - alert: LLMValidationWarnings
         expr: increase(llm_validation{type="analyst"}[30m]) > 5
         labels:
           severity: warning
         annotations:
           summary: "LLM validation warnings"
``` 

## Grafana dashboards
1. **Pipeline latency** — context builder, feature calc, total.
2. **Forecast Quality** — panels for MAE/SMAPE/DA per horizon.
3. **Agent performance** — weights by regime.
4. **Paper/Live P&L** — existing `trading_results.json` + new metrics from `paper_summary`.

## Logging
- JSON logs (`JSON_LOGS=1`), include `run_id` & `slot`.
- Export reasoning/critique to S3 for audits.
- Keep logs ≥90 days.

## Alerting flow
Alertmanager → Slack/Telegram (#ops-cryptoia). Define rotations, include on-call runbook link.

## Health checks
- `/health` endpoint for liveness.
- Smoke test job: daily `python services/pipeline/tests/integration/test_master_smoke.py` (TODO) to ensure pipeline executes.
