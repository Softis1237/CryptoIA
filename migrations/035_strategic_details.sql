-- Normalized details for SMC and Whale agents (optional, fast queries by fields)

CREATE TABLE IF NOT EXISTS smc_verdict_details (
  agent_name TEXT NOT NULL,
  symbol TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  entry_low DOUBLE PRECISION,
  entry_high DOUBLE PRECISION,
  invalidation DOUBLE PRECISION,
  target DOUBLE PRECISION,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (agent_name, symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_smc_details_symbol_ts ON smc_verdict_details (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS whale_verdict_details (
  agent_name TEXT NOT NULL,
  symbol TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  exchange_netflow DOUBLE PRECISION,
  whale_txs BIGINT,
  large_trades BIGINT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (agent_name, symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_whale_details_symbol_ts ON whale_verdict_details (symbol, ts DESC);

