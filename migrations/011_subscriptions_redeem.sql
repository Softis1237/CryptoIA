-- Redeem codes for subscriptions

CREATE TABLE IF NOT EXISTS redeem_codes (
  code TEXT PRIMARY KEY,
  months INT,
  invoice_id TEXT,
  used_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now()
);

