-- Technical analysis knowledge base: canonical pattern definitions
CREATE TABLE IF NOT EXISTS technical_patterns (
  pattern_id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  category TEXT NOT NULL,               -- candle|chart
  timeframe TEXT NOT NULL DEFAULT '1m', -- reference timeframe
  definition_json JSONB NOT NULL,       -- formalized definition/rules
  description TEXT,
  source TEXT,
  confidence_default REAL NOT NULL DEFAULT 0.6,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Seed a minimal set of common patterns
INSERT INTO technical_patterns (name, category, timeframe, definition_json, description, source, confidence_default)
VALUES
  ('hammer', 'candle', '1m', '{"type":"candle","lookback":1,"rules":{"lower_shadow_ge_body":"2.0","upper_shadow_le_body":"0.4"}}', 'Hammer: long lower wick, small body near high', 'seed', 0.6),
  ('shooting_star', 'candle', '1m', '{"type":"candle","lookback":1,"rules":{"upper_shadow_ge_body":"2.0","lower_shadow_le_body":"0.4"}}', 'Shooting Star: long upper wick, small body near low', 'seed', 0.6),
  ('engulfing_bull', 'candle', '1m', '{"type":"candle_pair","lookback":2,"rules":{"prev_red":true,"curr_green":true,"engulf":true}}', 'Bullish Engulfing: green candle engulfs previous red', 'seed', 0.65),
  ('engulfing_bear', 'candle', '1m', '{"type":"candle_pair","lookback":2,"rules":{"prev_green":true,"curr_red":true,"engulf":true}}', 'Bearish Engulfing: red candle engulfs previous green', 'seed', 0.65),
  ('morning_star', 'candle', '1m', '{"type":"candle_triplet","lookback":3,"rules":{"first_red":true,"middle_small":true,"third_green":true,"close_above_mid_first":true}}', 'Morning Star: reversal 3-candle formation', 'seed', 0.65),
  ('double_top', 'chart', '1m', '{"type":"chart","lookback":120,"rules":{"peaks":2,"tolerance_bps":50}}', 'Double Top in recent window', 'seed', 0.55),
  ('double_bottom', 'chart', '1m', '{"type":"chart","lookback":120,"rules":{"troughs":2,"tolerance_bps":50}}', 'Double Bottom in recent window', 'seed', 0.55),
  ('head_and_shoulders', 'chart', '1m', '{"type":"chart","lookback":240,"rules":{"structure":"HnS"}}', 'Head and Shoulders in recent window', 'seed', 0.55)
ON CONFLICT (name) DO NOTHING;

