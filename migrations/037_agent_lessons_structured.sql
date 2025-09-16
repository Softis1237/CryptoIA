-- Enforce structured metadata for agent lessons and prevent duplicates
ALTER TABLE agent_lessons
    ALTER COLUMN meta SET DATA TYPE JSONB USING COALESCE(meta, '{}'::jsonb),
    ALTER COLUMN meta SET DEFAULT '{}'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_lessons_scope_hash
    ON agent_lessons (scope, ((meta->>'hash')))
    WHERE meta ? 'hash';
