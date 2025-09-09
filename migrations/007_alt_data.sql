-- Alternative data signals table

CREATE TABLE IF NOT EXISTS alt_signals (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  run_id TEXT,
  ts TIMESTAMPTZ NOT NULL,
  source TEXT NOT NULL,
  metric TEXT NOT NULL,
  value NUMERIC,
  meta JSONB
);

CREATE INDEX IF NOT EXISTS idx_alt_ts ON alt_signals (ts);

