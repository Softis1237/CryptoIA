-- Compressed agent memory: concise lessons derived from run summaries
CREATE TABLE IF NOT EXISTS agent_lessons (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  scope TEXT NOT NULL DEFAULT 'global',
  lesson_text TEXT NOT NULL,
  meta JSONB
);

CREATE INDEX IF NOT EXISTS idx_agent_lessons_created ON agent_lessons (created_at DESC);

