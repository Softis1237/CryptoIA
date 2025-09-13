-- Metrics for discovered patterns validation
CREATE TABLE IF NOT EXISTS pattern_discovery_metrics (
  id SERIAL PRIMARY KEY,
  discovered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  window_hours INT NOT NULL,
  move_threshold REAL NOT NULL,
  sample_count INT NOT NULL,
  pattern_name TEXT NOT NULL,
  expected_direction TEXT NOT NULL,
  match_count INT NOT NULL,
  success_count INT NOT NULL,
  success_rate REAL NOT NULL,
  p_value REAL NOT NULL,
  definition_json JSONB,
  summary_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_pattern_discovery_time ON pattern_discovery_metrics (discovered_at DESC);

