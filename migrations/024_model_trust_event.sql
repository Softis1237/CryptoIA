-- Per-regime + event-type model trust weights
CREATE TABLE IF NOT EXISTS model_trust_regime_event (
  regime_label TEXT NOT NULL,
  horizon TEXT NOT NULL,
  event_type TEXT NOT NULL,
  model TEXT NOT NULL,
  weight DOUBLE PRECISION NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (regime_label, horizon, event_type, model)
);
CREATE INDEX IF NOT EXISTS idx_mtre_reg_hz_evt ON model_trust_regime_event (regime_label, horizon, event_type);

