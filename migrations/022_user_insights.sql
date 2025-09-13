-- User-submitted insights
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS user_insights (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id BIGINT NOT NULL,
  text TEXT NOT NULL,
  url TEXT,
  verdict TEXT,
  score_truth NUMERIC,
  score_freshness NUMERIC,
  meta JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_insights_created ON user_insights (created_at DESC);
