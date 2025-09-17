-- Portfolio state and risk exposure (optional)

CREATE TABLE IF NOT EXISTS portfolio_positions (
  symbol TEXT PRIMARY KEY,
  direction TEXT NOT NULL,
  quantity DOUBLE PRECISION NOT NULL,
  avg_price DOUBLE PRECISION NOT NULL,
  unrealized_pnl DOUBLE PRECISION,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  closed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS portfolio_trades (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity DOUBLE PRECISION NOT NULL,
  price DOUBLE PRECISION NOT NULL,
  fee DOUBLE PRECISION,
  meta JSONB
);

CREATE TABLE IF NOT EXISTS portfolio_risk_exposure (
  as_of TIMESTAMPTZ NOT NULL DEFAULT now(),
  symbol TEXT NOT NULL,
  potential_loss DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (as_of, symbol)
);

CREATE TABLE IF NOT EXISTS portfolio_equity_daily (
  date DATE PRIMARY KEY,
  equity DOUBLE PRECISION NOT NULL
);

