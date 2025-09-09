-- Social, futures and on-chain signals tables

CREATE TABLE IF NOT EXISTS social_signals (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  src_id TEXT UNIQUE,
  ts TIMESTAMPTZ NOT NULL,
  platform TEXT NOT NULL,
  author TEXT,
  text TEXT,
  url TEXT,
  sentiment TEXT,
  score NUMERIC,
  topics TEXT[],
  metrics JSONB
);

CREATE INDEX IF NOT EXISTS idx_social_ts ON social_signals (ts);
CREATE INDEX IF NOT EXISTS idx_social_platform ON social_signals (platform);

CREATE TABLE IF NOT EXISTS onchain_signals (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  run_id TEXT,
  ts TIMESTAMPTZ NOT NULL,
  asset TEXT NOT NULL,
  metric TEXT NOT NULL,
  value NUMERIC,
  interpretation TEXT
);

CREATE INDEX IF NOT EXISTS idx_onchain_ts ON onchain_signals (ts);

CREATE TABLE IF NOT EXISTS futures_metrics (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  run_id TEXT,
  ts TIMESTAMPTZ NOT NULL,
  exchange TEXT,
  symbol TEXT,
  funding_rate NUMERIC,
  next_funding_time TIMESTAMPTZ,
  mark_price NUMERIC,
  index_price NUMERIC,
  open_interest NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_futures_ts ON futures_metrics (ts);

