-- 044_arbiter_reasoning.sql
-- Логи цепочки рассуждений и самокритики арбитра

CREATE TABLE IF NOT EXISTS arbiter_reasoning (
    run_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    mode TEXT NOT NULL,
    context_ref TEXT,
    tokens_estimate NUMERIC,
    analysis JSONB NOT NULL,
    context JSONB,
    s3_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_arbiter_reasoning_created_at
    ON arbiter_reasoning (created_at DESC);

CREATE TABLE IF NOT EXISTS arbiter_selfcritique (
    run_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    recommendation TEXT,
    probability_delta NUMERIC,
    critique JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_arbiter_selfcritique_created_at
    ON arbiter_selfcritique (created_at DESC);
