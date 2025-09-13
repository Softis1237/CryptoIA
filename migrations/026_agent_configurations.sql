-- Agent configurations: prompts and parameters per agent with versions
CREATE TABLE IF NOT EXISTS agent_configurations (
  id SERIAL PRIMARY KEY,
  agent_name TEXT NOT NULL,
  version INT NOT NULL,
  system_prompt TEXT,
  parameters_json JSONB,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(agent_name, version)
);

