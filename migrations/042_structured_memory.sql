-- Structured lessons storage for MemoryGuardian

CREATE TABLE IF NOT EXISTS agent_lessons_structured (
    id BIGSERIAL PRIMARY KEY,
    scope TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    lesson JSONB NOT NULL,
    error_type TEXT,
    market_regime TEXT,
    involved_agents TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    triggering_signals TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    key_factors_missed TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    correct_action_suggestion TEXT,
    confidence_before DOUBLE PRECISION,
    outcome_after DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lesson_embedding vector(1536)
);

CREATE INDEX IF NOT EXISTS idx_lessons_structured_scope ON agent_lessons_structured (scope);
CREATE INDEX IF NOT EXISTS idx_lessons_structured_regime ON agent_lessons_structured (market_regime);

DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_lessons_structured_embedding
        ON agent_lessons_structured USING ivfflat (lesson_embedding vector_cosine_ops) WITH (lists = 100);
EXCEPTION WHEN others THEN
    NULL;
END $$;

