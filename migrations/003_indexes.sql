-- Useful indexes for runtime performance
CREATE INDEX IF NOT EXISTS idx_predictions_run_horizon ON predictions (run_id, horizon);
CREATE INDEX IF NOT EXISTS idx_news_signals_ts ON news_signals (ts);
CREATE INDEX IF NOT EXISTS idx_paper_equity_curve_ts ON paper_equity_curve (ts);
CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions (user_id);

