-- Enforce structured metadata for agent lessons and prevent duplicates
ALTER TABLE agent_lessons
    ALTER COLUMN meta SET DATA TYPE JSONB USING COALESCE(meta, '{}'::jsonb),
    ALTER COLUMN meta SET DEFAULT '{}'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_lessons_scope_hash
    ON agent_lessons (scope, ((meta->>'hash')))
    WHERE meta ? 'hash';

-- Structured storage for agent lessons and fallback quality metrics
ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS lesson JSONB;
ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS insight TEXT;
ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS action TEXT;
ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS risk TEXT;

CREATE TABLE IF NOT EXISTS agent_lesson_metrics (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  scope TEXT NOT NULL DEFAULT 'global',
  mode TEXT NOT NULL DEFAULT 'unknown',
  rows_processed INT NOT NULL,
  lessons_inserted INT NOT NULL,
  metrics JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_agent_lesson_metrics_created ON agent_lesson_metrics (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_lessons_scope ON agent_lessons (scope);
CREATE INDEX IF NOT EXISTS idx_agent_lessons_title ON agent_lessons (title);
