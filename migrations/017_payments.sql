-- Payments table

CREATE TABLE IF NOT EXISTS payments (
  charge_id TEXT PRIMARY KEY,
  user_id BIGINT NOT NULL,
  amount BIGINT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
