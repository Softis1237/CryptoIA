-- Agents predictions and metrics storage

CREATE TABLE IF NOT EXISTS agents_predictions (
  run_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  result_json JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (run_id, agent)
);

CREATE TABLE IF NOT EXISTS agents_metrics (
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  agent TEXT NOT NULL,
  metric TEXT NOT NULL,
  value NUMERIC,
  labels JSONB,
  PRIMARY KEY (ts, agent, metric)
);

CREATE INDEX IF NOT EXISTS idx_agents_predictions_run ON agents_predictions (run_id);
CREATE INDEX IF NOT EXISTS idx_agents_metrics_agent_metric ON agents_metrics (agent, metric);

