-- Knowledge Core documents (embeddings via pgvector)
CREATE TABLE IF NOT EXISTS knowledge_docs (
  doc_id TEXT PRIMARY KEY,
  source TEXT,
  title TEXT,
  chunk_index INT,
  content TEXT,
  dim INT NOT NULL,
  embedding VECTOR,
  meta JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Optional index for ANN (ivfflat) can be created when data grows and lists tuned:
-- CREATE INDEX IF NOT EXISTS idx_knowledge_docs_ann ON knowledge_docs USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

