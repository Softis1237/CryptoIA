-- Embeddings for agent lessons (pgvector)

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE agent_lessons
    ADD COLUMN IF NOT EXISTS lesson_embedding vector(1536);

-- IVF index for fast ANN search (optional, requires pgvector >= 0.4)
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_agent_lessons_embedding
        ON agent_lessons USING ivfflat (lesson_embedding vector_cosine_ops) WITH (lists = 100);
EXCEPTION WHEN others THEN
    -- fallback: skip index if ivfflat not available
    NULL;
END $$;

