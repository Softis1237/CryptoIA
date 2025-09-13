-- Affiliates and referrals

CREATE TABLE IF NOT EXISTS affiliates (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  partner_user_id BIGINT UNIQUE NOT NULL,
  partner_name TEXT,
  code TEXT UNIQUE NOT NULL,
  percent INT NOT NULL DEFAULT 50,
  balance BIGINT NOT NULL DEFAULT 0, -- in smallest currency unit (e.g., stars)
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS referrals (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  partner_user_id BIGINT NOT NULL,
  referred_user_id BIGINT NOT NULL,
  code TEXT,
  charge_id TEXT,
  amount BIGINT NOT NULL,
  commission BIGINT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Users: add referrer info
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS referrer_code TEXT,
  ADD COLUMN IF NOT EXISTS referrer_name TEXT;

CREATE INDEX IF NOT EXISTS idx_referrals_partner ON referrals (partner_user_id);
CREATE INDEX IF NOT EXISTS idx_referrals_referred ON referrals (referred_user_id);
CREATE INDEX IF NOT EXISTS idx_affiliates_code ON affiliates (code);

