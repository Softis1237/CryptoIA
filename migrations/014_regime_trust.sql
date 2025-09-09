-- Per-regime model trust weights (to modulate ensemble)

CREATE TABLE IF NOT EXISTS model_trust_regime (
  regime_label TEXT NOT NULL,
  horizon TEXT NOT NULL,
  model TEXT NOT NULL,
  weight NUMERIC NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (regime_label, horizon, model)
);

CREATE INDEX IF NOT EXISTS idx_model_trust_regime_updated ON model_trust_regime (updated_at);

