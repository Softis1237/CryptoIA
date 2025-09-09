-- Live trading tables (optional)

CREATE TABLE IF NOT EXISTS live_orders (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ DEFAULT now(),
  exchange TEXT,
  symbol TEXT,
  side TEXT,
  type TEXT,
  amount NUMERIC,
  price NUMERIC,
  params JSONB,
  status TEXT,
  exchange_order_id TEXT,
  info JSONB
);

CREATE INDEX IF NOT EXISTS idx_live_orders_exchange ON live_orders (exchange);
CREATE INDEX IF NOT EXISTS idx_live_orders_symbol ON live_orders (symbol);

