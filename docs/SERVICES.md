Services Overview

This project runs as a Docker Compose stack. Here is what each service does, how to access it, and which secrets or keys it may need.

Core runtime

- pipeline: Python code for data ingestion, features, models, ensembles, risk and publishing. Usually invoked by the coordinator on schedule, but you can run tasks ad‑hoc via docker compose run.
- coordinator: Scheduler loop that triggers the twice‑daily forecast pipeline (00:00 and 12:00 in TIMEZONE). Also pushes metrics and optionally retrains models.
- bot: Telegram bot for subscriptions (Stars), direct messages, promo codes, and user insights collection. It can publish forecasts either to a channel or as DMs.

Storage and cache

- postgres: Primary database (tables for subscriptions, payments, predictions, metrics, etc.).
- redis: Cache for LLM responses and lightweight locks.
- minio: S3‑compatible object storage for artifacts (features/parquet, charts, models). Console at http://localhost:9001 (defaults are in .env).

Observability

- pushgateway: Receives metrics from the pipeline (push model).
- prometheus: Scrapes pushgateway and serves time‑series metrics (UI at http://localhost:9094 when ports are exposed).
- grafana: Dashboards (UI at http://localhost:3003 when ports are exposed). Default admin/admin.

LLM orchestration

- flowise: Visual builder for LLM flows (UI at http://localhost:3002 when ports are exposed). Create flows for: sentiment, forecast helper, explain, debate (arbiter), scenario, validate. Each flow exposes an API endpoint: POST /api/v1/prediction/{flowId}.
- mcp (mini): Lightweight HTTP server exposing safe tools to LLMs (regime detection, similarity, models+ensemble, recent run memory). Auto-started in scheduler when ENABLE_MCP=1.
  - Extra tools: `run_event_study`, `run_pattern_discovery` (dry-run by default), `compress_memory`, `get_lessons`, `run_cognitive_architect`.

Optional helpers

- windmill: Low‑code runner (kept for future ops automations). UI at http://localhost:8000 (already exposed).
- order flow (in‑pipeline): optional collector that fetches recent trades via CCXT and enriches features with OFI/дельта. Enable with ENABLE_ORDER_FLOW=1.
- pattern discovery (Airflow): weekly DAG `pattern_discovery_weekly` runs PatternDiscoveryAgent to propose new entries for `technical_patterns` (by default in dry‑run mode).
- memory compress (Airflow): monthly DAG `memory_compress_monthly` aggregates recent run summaries into concise lessons (`agent_lessons` table).
- cognitive architect (Airflow): monthly DAG `cognitive_architect_monthly` proposes updated prompts/configs; writes new versions to `agent_configurations`.

ChartReasoningAgent

- Integrated into MasterAgent pipeline after features build. Uses `agent_configurations` (agent_name="ChartReasoningAgent") for prompt/params; fallback to defaults.
- Context includes: ATR, MACD, BB width, VWAP, local high/low (50), ATR band, quantiles (Q10/50/90), pivots (P/R1/S1), and detected patterns (pat_*).

MCP examples (inside docker network)

```
curl -s http://coordinator:8765/call -X POST -H 'Content-Type: application/json' \
  -d '{"tool":"run_pattern_discovery","params":{"timeframe":"1h","days":60,"dry_run":true}}'

curl -s http://coordinator:8765/call -X POST -H 'Content-Type: application/json' \
  -d '{"tool":"compress_memory","params":{"n":50,"scope":"global"}}'

curl -s http://coordinator:8765/call -X POST -H 'Content-Type: application/json' \
  -d '{"tool":"get_lessons","params":{"n":5}}'

curl -s http://coordinator:8765/call -X POST -H 'Content-Type: application/json' \
  -d '{"tool":"run_cognitive_architect","params":{"analyze_n":50}}'
```

Keys and where to get them

- Telegram Bot Token: Create bot in @BotFather → /newbot → copy token. If token is leaked or rejected (Unauthorized), do /revoke and /token to issue a new one. Set TELEGRAM_BOT_TOKEN in .env.
- OpenAI API Key: https://platform.openai.com → Create API key. Set OPENAI_API_KEY in .env. Used for news facts extraction and fallbacks.
- Dune API Key (free tier is enough for on‑chain): https://dune.com → Profile → API Keys. Set DUNE_API_KEY in .env and configure query IDs via DUNE_QUERY_* variables.
- News API keys (optional): CRYPTOPANIC_TOKEN, NEWSAPI_KEY. Without them project falls back to RSS.
- Flowise API: Flow endpoints live at http://flowise:3000/api/v1/prediction/{flowId} inside Docker network. No extra key is required by default (Flowise username/password protect the UI; keep API internal and don’t expose outside Docker in production). If you enable API keys in Flowise, you’ll need to add an Authorization header in call_flowise_json (not enabled in this repo).

Configuring Flowise flows

1) Open Flowise UI at http://localhost:3002 (user/pass from docker-compose). Create a Chatflow for each function:
   - Sentiment: input (compact headlines) → LLM → JSON output {sentiment, impact_score, topics}
   - Forecast helper (optional): input (news/regime/ATR) → LLM → JSON {y_hat, p_up}
   - Explain (bullets), Debate (bullets+risk_flags), Scenario (5 branches), Validate (verdict/scores)
2) Get each flowId from the Flowise UI (Details → API). Set env vars to the full endpoint:
   - FLOWISE_SENTIMENT_URL=http://flowise:3000/api/v1/prediction/<SENTIMENT_FLOW_ID>
   - FLOWISE_FORECAST_URL=http://flowise:3000/api/v1/prediction/<FORECAST_FLOW_ID>
   - FLOWISE_EXPLAIN_URL=http://flowise:3000/api/v1/prediction/<EXPLAIN_FLOW_ID>
   - FLOWISE_DEBATE_URL=http://flowise:3000/api/v1/prediction/<DEBATE_FLOW_ID>
   - FLOWISE_SCENARIO_URL=http://flowise:3000/api/v1/prediction/<SCENARIO_FLOW_ID>
   - FLOWISE_VALIDATE_URL=http://flowise:3000/api/v1/prediction/<VALIDATE_FLOW_ID>

Minimum viable configuration

- Telegram: TELEGRAM_BOT_TOKEN, TELEGRAM_DM_USER_IDS (numeric IDs). Each recipient must /start the bot once.
- No paid news keys required: RSS fallback is enabled and sentiment heuristics work. Flowise endpoints can be left blank at first; LLM steps will be skipped with warnings.
- S3/MinIO and Postgres are provisioned by docker-compose.

MasterAgent (dynamic orchestration)

- Run once: `docker compose run --rm pipeline python -m pipeline.agents.master --slot=manual`
- Environment:
  - `USE_MASTER_AGENT=1` to switch your own scheduler/runner to MasterAgent (default in `.env.example`).
  - `OPENAI_MODEL_MASTER=gpt-4o` for complex reasoning, `LLM_MAX_TOKENS=4096` for ample headroom.
  - `ENABLE_MULTI_DEBATE=1` to enrich analysis with bull/bear/quant personas.

MCP tools available

- `run_regime_detection(features_s3)` → `{label, confidence, features}`
- `run_similarity_search(features_s3)` → `{topk: [{period, distance}, ...]}`
- `run_models_and_ensemble(features_s3, horizon)` → `ensemble, preds, last_price, atr`
- `get_recent_run_summaries(n)` → recent memory items for context
- `run_event_study(event_type, k?, window_hours?, symbol?, provider?)` → basic study summary
- `run_pattern_discovery(...)` → stub (returns `{status: 'not-implemented'}` in Phase 1)

Health and validation

- Health endpoints: coordinator and bot run an internal HTTP health check at http://127.0.0.1:8000/healthz inside the container; docker-compose restarts containers on failures.
- Quick pipeline test:
  docker compose run --rm pipeline python -m pipeline.orchestration.agent_flow --slot=manual
  Expect DM with text and chart; artifacts under MinIO bucket artifacts/runs/<date>/manual.

Common issues

- Telegram InvalidToken: revoke token in @BotFather and paste the new one to .env; restart bot; delete_webhook as needed.
- Ports already in use: change host ports in docker-compose for grafana/prometheus/flowise/pushgateway.
- Flowise API unset: LLM calls are skipped with warnings; the rest of the pipeline still runs.
