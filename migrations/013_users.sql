-- Users table (preferences)

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id BIGINT UNIQUE,
  email TEXT,
  tz TEXT,
  assets TEXT[],
  prefs JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

