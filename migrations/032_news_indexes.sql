-- Indexes to speed up real-time NEWS scanning
CREATE INDEX IF NOT EXISTS idx_news_signals_ts ON news_signals (ts);
CREATE INDEX IF NOT EXISTS idx_news_signals_impact ON news_signals (impact_score);

