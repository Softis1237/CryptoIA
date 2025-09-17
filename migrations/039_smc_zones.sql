-- SMC zones storage (multi-timeframe, normalized)

CREATE TABLE IF NOT EXISTS smc_zones (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  zone_type TEXT NOT NULL, -- OB_BULL, OB_BEAR, FVG, LIQUIDITY_POOL, BREAKER
  price_low DOUBLE PRECISION,
  price_high DOUBLE PRECISION,
  status TEXT,            -- untested, mitigated, invalidated
  meta JSONB
);

CREATE INDEX IF NOT EXISTS idx_smc_zones_symbol_tf ON smc_zones (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_smc_zones_type ON smc_zones (zone_type);

