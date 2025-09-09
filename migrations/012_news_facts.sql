-- LLM-extracted news facts storage

CREATE TABLE IF NOT EXISTS news_facts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  src_id TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  type TEXT NOT NULL,
  direction TEXT NOT NULL,
  magnitude NUMERIC,
  confidence NUMERIC,
  entities JSONB,
  raw JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (src_id, type, ts)
);

CREATE INDEX IF NOT EXISTS idx_news_facts_ts ON news_facts (ts);
CREATE INDEX IF NOT EXISTS idx_news_facts_src ON news_facts (src_id);

