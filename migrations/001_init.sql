-- Базовые схемы согласно ТЗ (MVP-срез)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Цены/агрегации
CREATE TABLE IF NOT EXISTS prices_agg (
  ts TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  open NUMERIC,
  high NUMERIC,
  low NUMERIC,
  close NUMERIC,
  volume NUMERIC,
  provider TEXT,
  PRIMARY KEY (ts, symbol)
);

-- Снимок фичей
CREATE TABLE IF NOT EXISTS features_snapshot (
  run_id TEXT PRIMARY KEY,
  ts_window TIMESTAMPTZ NOT NULL,
  path_s3 TEXT NOT NULL
);

-- Режим рынка
CREATE TABLE IF NOT EXISTS regimes (
  run_id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  confidence NUMERIC,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Похожие окна
CREATE TABLE IF NOT EXISTS similar_windows (
  run_id TEXT PRIMARY KEY,
  topk_json JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Предсказания
CREATE TABLE IF NOT EXISTS predictions (
  run_id TEXT NOT NULL,
  horizon TEXT NOT NULL,
  y_hat NUMERIC,
  pi_low NUMERIC,
  pi_high NUMERIC,
  proba_up NUMERIC,
  per_model_json JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (run_id, horizon)
);

-- Веса ансамбля
CREATE TABLE IF NOT EXISTS ensemble_weights (
  run_id TEXT NOT NULL,
  model TEXT NOT NULL,
  w NUMERIC NOT NULL,
  PRIMARY KEY (run_id, model)
);

-- Объяснения/флаги риска
CREATE TABLE IF NOT EXISTS explanations (
  run_id TEXT PRIMARY KEY,
  markdown TEXT,
  risk_flags JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Сценарии
CREATE TABLE IF NOT EXISTS scenarios (
  run_id TEXT PRIMARY KEY,
  list_json JSONB NOT NULL,
  charts_s3 TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Рекомендации/карточки сделок
CREATE TABLE IF NOT EXISTS trades_suggestions (
  run_id TEXT PRIMARY KEY,
  side TEXT NOT NULL,
  entry_zone JSONB,
  leverage NUMERIC,
  sl NUMERIC,
  tp NUMERIC,
  rr NUMERIC,
  reason_codes JSONB,
  times_json JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Лог ошибок
CREATE TABLE IF NOT EXISTS errors_log (
  run_id TEXT,
  horizon TEXT,
  metrics_json JSONB,
  regime TEXT,
  features_digest TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Новости/сигналы
CREATE TABLE IF NOT EXISTS news_signals (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  src_id TEXT UNIQUE,
  ts TIMESTAMPTZ NOT NULL,
  title TEXT NOT NULL,
  source TEXT,
  url TEXT,
  sentiment TEXT,
  topics TEXT[],
  impact_score NUMERIC
);

-- Бумажная торговля
CREATE TABLE IF NOT EXISTS paper_accounts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  start_equity NUMERIC NOT NULL,
  equity NUMERIC NOT NULL,
  cfg_json JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paper_positions (
  pos_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  account_id UUID REFERENCES paper_accounts(id),
  opened_at TIMESTAMPTZ NOT NULL,
  side TEXT NOT NULL,
  entry NUMERIC,
  leverage NUMERIC,
  qty NUMERIC,
  sl NUMERIC,
  tp NUMERIC,
  status TEXT,
  meta_json JSONB
);

CREATE TABLE IF NOT EXISTS paper_orders (
  order_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  pos_id UUID REFERENCES paper_positions(pos_id),
  type TEXT,
  price NUMERIC,
  qty NUMERIC,
  state TEXT,
  expires_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS paper_trades (
  trade_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  pos_id UUID REFERENCES paper_positions(pos_id),
  ts TIMESTAMPTZ,
  price NUMERIC,
  qty NUMERIC,
  side TEXT,
  fee NUMERIC,
  reason TEXT
);

CREATE TABLE IF NOT EXISTS paper_pnl (
  pos_id UUID PRIMARY KEY REFERENCES paper_positions(pos_id),
  realized_pnl NUMERIC,
  rr NUMERIC,
  mae NUMERIC,
  mfe NUMERIC,
  elapsed_s BIGINT
);

CREATE TABLE IF NOT EXISTS paper_equity_curve (
  ts TIMESTAMPTZ NOT NULL,
  account_id UUID REFERENCES paper_accounts(id),
  equity NUMERIC NOT NULL,
  PRIMARY KEY (ts, account_id)
);
