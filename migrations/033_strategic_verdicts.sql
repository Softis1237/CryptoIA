-- Strategic agent verdicts (SMC Analyst, Whale Watcher, etc.)
CREATE TABLE IF NOT EXISTS strategic_verdicts (
  agent_name TEXT NOT NULL,
  symbol TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  verdict TEXT NOT NULL,
  confidence DOUBLE PRECISION,
  meta JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (agent_name, symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_strategic_verdicts_latest
  ON strategic_verdicts (agent_name, symbol, ts DESC);

