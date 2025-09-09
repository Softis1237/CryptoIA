-- Extend news_signals with confidence and add social_index (pgvector)

ALTER TABLE IF EXISTS news_signals
  ADD COLUMN IF NOT EXISTS confidence NUMERIC;

-- pgvector table for social embeddings
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS social_index (
  src_id TEXT PRIMARY KEY,
  platform TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  dim INT NOT NULL,
  embedding vector,
  meta JSONB
);

CREATE INDEX IF NOT EXISTS social_index_ivfflat ON social_index USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

