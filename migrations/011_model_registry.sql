-- Simple model registry

CREATE TABLE IF NOT EXISTS model_registry (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  path_s3 TEXT,
  params JSONB,
  metrics JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (name, version)
);

