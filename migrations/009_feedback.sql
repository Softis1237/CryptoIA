-- Users feedback table for learning on feedback

CREATE TABLE IF NOT EXISTS users_feedback (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id BIGINT NOT NULL,
  run_id TEXT,
  rating SMALLINT,
  comment TEXT,
  meta JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_user ON users_feedback (user_id);

