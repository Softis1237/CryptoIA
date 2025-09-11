-- Realized outcomes for predictions (to drive feedback loop)

CREATE TABLE IF NOT EXISTS prediction_outcomes (
  run_id TEXT NOT NULL,
  horizon TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  y_hat NUMERIC,
  y_true NUMERIC,
  error_abs NUMERIC,
  error_pct NUMERIC,
  direction_correct BOOLEAN,
  regime_label TEXT,
  news_ctx NUMERIC,
  tags JSONB,
  PRIMARY KEY (run_id, horizon)
);

CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_created ON prediction_outcomes (created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_hz ON prediction_outcomes (horizon);

