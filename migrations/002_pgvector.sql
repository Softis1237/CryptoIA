-- pgvector extension and features index
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS features_index (
  ts_window TIMESTAMPTZ PRIMARY KEY,
  symbol TEXT NOT NULL,
  dim INT NOT NULL,
  embedding vector(16) NOT NULL,
  meta JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for ANN search (requires Postgres >= 14 with pgvector)
CREATE INDEX IF NOT EXISTS features_index_ivfflat ON features_index USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

