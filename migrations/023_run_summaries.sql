-- Summaries of each release run for agent memory
CREATE TABLE IF NOT EXISTS run_summaries (
  run_id TEXT PRIMARY KEY,
  final_analysis_json JSONB,
  prediction_outcome_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_run_summaries_created ON run_summaries (created_at DESC);

