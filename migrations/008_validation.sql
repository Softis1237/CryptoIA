-- Validation and backtest storage

CREATE TABLE IF NOT EXISTS validation_reports (
  run_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  items JSONB,
  warnings JSONB,
  errors JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS backtest_results (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  cfg_json JSONB,
  metrics_json JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

