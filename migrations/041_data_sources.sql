-- Strategic data sources lifecycle storage

CREATE TABLE IF NOT EXISTS data_sources (
    name TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    provider TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    popularity DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    reputation DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    trust_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_data_sources_provider ON data_sources (provider);
CREATE INDEX IF NOT EXISTS idx_data_sources_tags ON data_sources USING GIN (tags);

CREATE TABLE IF NOT EXISTS data_source_history (
    id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL REFERENCES data_sources(name) ON DELETE CASCADE,
    delta DOUBLE PRECISION NOT NULL,
    new_score DOUBLE PRECISION NOT NULL,
    reason TEXT NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_data_source_history_name ON data_source_history (source_name);

CREATE TABLE IF NOT EXISTS data_source_anomalies (
    id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL REFERENCES data_sources(name) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_data_source_anomalies_name ON data_source_anomalies (source_name);

CREATE TABLE IF NOT EXISTS data_source_tasks (
    id BIGSERIAL PRIMARY KEY,
    summary TEXT NOT NULL,
    description TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'medium',
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

