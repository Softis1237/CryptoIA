-- 045_agent_performance.sql
-- Track agent performance per regime for dynamic weighting

CREATE TABLE IF NOT EXISTS agent_performance (
    agent_name TEXT NOT NULL,
    regime_label TEXT NOT NULL,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    weight NUMERIC NOT NULL DEFAULT 1.0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (agent_name, regime_label)
);

CREATE INDEX IF NOT EXISTS idx_agent_performance_regime
    ON agent_performance (regime_label);
