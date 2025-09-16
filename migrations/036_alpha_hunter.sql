-- Alpha Hunter: elite trades, snapshots, strategies

CREATE TABLE IF NOT EXISTS elite_leaderboard_trades (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source TEXT NOT NULL,            -- e.g., binance, bybit
  trader_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,              -- LONG/SHORT
  entry_price DOUBLE PRECISION,
  ts TIMESTAMPTZ NOT NULL,
  pnl DOUBLE PRECISION,
  meta JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_elite_trades_ts ON elite_leaderboard_trades (ts DESC);
CREATE INDEX IF NOT EXISTS idx_elite_trades_trader ON elite_leaderboard_trades (trader_id);

CREATE TABLE IF NOT EXISTS alpha_snapshots (
  snapshot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  trade_id UUID REFERENCES elite_leaderboard_trades(id) ON DELETE CASCADE,
  context_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS alpha_strategies (
  strategy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT UNIQUE NOT NULL,
  definition_json JSONB NOT NULL,
  backtest_metrics_json JSONB,
  status TEXT NOT NULL DEFAULT 'active',   -- active | paused
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

