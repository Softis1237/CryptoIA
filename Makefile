SHELL := /bin/bash

.PHONY: up down logs rebuild precommit

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

precommit:
	pre-commit install

pipeline-shell:
	docker compose run --rm pipeline bash


analyze:
	python -m pipeline.cli.analysis $(ARGS)

.PHONY: migrate migrate-all seed-configs

# Apply single migration file: make migrate MIG=033_strategic_verdicts.sql
migrate:
	@if [ -z "$(MIG)" ]; then echo "Usage: make migrate MIG=033_strategic_verdicts.sql"; exit 2; fi
	@echo "Applying migration $(MIG)"
	@cat migrations/$(MIG) | docker compose exec -T postgres \
		psql -v ON_ERROR_STOP=1 -U $$POSTGRES_USER -d $$POSTGRES_DB -f -

# Apply all migrations in order
migrate-all:
	@for f in $$(ls -1 migrations/*.sql | sort); do \
		echo "Applying $$f"; \
		cat $$f | docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U $$POSTGRES_USER -d $$POSTGRES_DB -f - || exit 1; \
	done

# Seed default agent configurations (AlertPriority / ConfidenceAggregator)
seed-configs:
	@PYTHONPATH=services/pipeline/src python - <<'PY'
from pipeline.infra.db import insert_agent_config
alert_params = {
  "weights": {
    "VOL_SPIKE": 1.0, "DELTA_SPIKE": 1.0, "ATR_SPIKE": 1.2, "MOMENTUM": 1.3,
    "NEWS": 1.4, "L2_WALL": 0.8, "L2_IMBALANCE": 1.0,
    "PATTERN_BULL_ENGULF": 1.2, "PATTERN_BEAR_ENGULF": 1.2,
    "PATTERN_HAMMER": 1.0, "PATTERN_SHOOTING_STAR": 1.0,
    "DERIV_OI_JUMP": 1.2, "DERIV_FUNDING": 1.1
  },
  "thresholds": {"low": 0.0, "medium": 1.0, "high": 2.0, "critical": 3.0},
  "critical_combos": [["MOMENTUM","VOL_SPIKE"],["MOMENTUM","PATTERN_BULL_ENGULF"],["MOMENTUM","PATTERN_BEAR_ENGULF"]]
}
conf_params = {"w_trigger":0.08,"w_pattern":0.10,"w_deriv":0.08,"w_ta":0.08,"w_regime":0.05,"w_news":0.05,"w_alert":0.05,"w_smc":0.10,"w_whale":0.08}
v1 = insert_agent_config("AlertPriority", None, alert_params, make_active=True)
v2 = insert_agent_config("ConfidenceAggregator", None, conf_params, make_active=True)
	print({"AlertPriority": v1, "ConfidenceAggregator": v2})
	PY

.PHONY: e2e-up e2e-alpha e2e-down

e2e-up:
	docker compose up -d postgres redis minio createbuckets pipeline rt_master trigger_agent whale_stream onchain_webhook
	@echo "Waiting services to be healthy..." && sleep 5

e2e-alpha:
	# Run Alpha Hunter with synthetic sample (dry_run)
	PYTHONPATH=services/pipeline/src python -m pipeline.agents.alpha_hunter '{"symbol":"BTC/USDT","dry_run":true}'
	# Build small knowledge index from docs (if exists)
	PYTHONPATH=services/pipeline/src python - <<'PY'
from pipeline.mcp.client import call_tool
import os
doc = 'docs/TRADING_PRIMER.md'
if os.path.exists(doc):
    print(call_tool('knowledge_build', {'sources':[doc]}))
print(call_tool('knowledge_query', {'query':'bullish engulfing', 'top_k':2}))
PY

e2e-down:
	docker compose down -v
