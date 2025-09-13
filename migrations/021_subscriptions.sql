-- Subscriptions table (ensure exists via migrations, not only runtime)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS subscriptions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id BIGINT NOT NULL,
  provider TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ends_at TIMESTAMPTZ NOT NULL,
  payload JSONB,
  UNIQUE (user_id, status)
);

-- Useful index
CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions (user_id);
