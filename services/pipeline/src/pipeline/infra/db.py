from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Tuple

import psycopg2
from loguru import logger
from psycopg2.extras import Json

from .config import settings


def _dsn() -> str:
    return (
        f"dbname={settings.postgres_db} user={settings.postgres_user} "
        f"password={settings.postgres_password} host={settings.postgres_host} port={settings.postgres_port}"
    )


@contextmanager
def get_conn():
    conn = psycopg2.connect(_dsn())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============ Affiliates / Referrals ============

def ensure_affiliates_tables() -> None:
    sql_aff = (
        "CREATE TABLE IF NOT EXISTS affiliates (\n"
        "  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),\n"
        "  partner_user_id BIGINT UNIQUE NOT NULL,\n"
        "  partner_name TEXT,\n"
        "  code TEXT UNIQUE NOT NULL,\n"
        "  percent INT NOT NULL DEFAULT 50,\n"
        "  balance BIGINT NOT NULL DEFAULT 0,\n"
        "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n"
        ")"
    )
    sql_ref = (
        "CREATE TABLE IF NOT EXISTS referrals (\n"
        "  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),\n"
        "  partner_user_id BIGINT NOT NULL,\n"
        "  referred_user_id BIGINT NOT NULL,\n"
        "  code TEXT,\n"
        "  charge_id TEXT,\n"
        "  amount BIGINT NOT NULL,\n"
        "  commission BIGINT NOT NULL,\n"
        "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n"
        ")"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_aff)
            cur.execute(sql_ref)


def _gen_ref_code(user_id: int) -> str:
    # Simple code from base36 of user_id
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = int(user_id)
    out = []
    if n == 0:
        return "A0"
    while n > 0:
        n, r = divmod(n, 36)
        out.append(chars[r])
    return "A" + "".join(reversed(out))


def get_or_create_affiliate(partner_user_id: int, partner_name: str | None = None, percent: int = 50) -> tuple[str, int]:
    """Return (code, percent) for partner; create if not exists."""
    ensure_affiliates_tables()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT code, percent FROM affiliates WHERE partner_user_id=%s",
                (partner_user_id,),
            )
            row = cur.fetchone()
            if row:
                return str(row[0]), int(row[1])
            code = _gen_ref_code(partner_user_id)
            try:
                cur.execute(
                    "INSERT INTO affiliates (partner_user_id, partner_name, code, percent) VALUES (%s, %s, %s, %s)",
                    (partner_user_id, partner_name, code, percent),
                )
            except Exception:
                # Fallback: random suffix collision
                code = f"A{partner_user_id}"
                cur.execute(
                    "INSERT INTO affiliates (partner_user_id, partner_name, code, percent) VALUES (%s, %s, %s, %s)",
                    (partner_user_id, partner_name, code, percent),
                )
            return code, percent


def set_affiliate_percent(partner_user_id: int, percent: int) -> None:
    ensure_affiliates_tables()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE affiliates SET percent=%s WHERE partner_user_id=%s",
                (percent, partner_user_id),
            )


def get_affiliate_by_code(code: str) -> tuple[int, int, str | None] | None:
    ensure_affiliates_tables()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT partner_user_id, percent, partner_name FROM affiliates WHERE code=%s",
                (code,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return int(row[0]), int(row[1]), (row[2] if row[2] is not None else None)


def upsert_user_referrer(user_id: int, ref_code: str, ref_name: str | None = None) -> None:
    sql = (
        "INSERT INTO users (user_id, referrer_code, referrer_name) VALUES (%s, %s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET referrer_code=EXCLUDED.referrer_code, referrer_name=EXCLUDED.referrer_name"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, ref_code, ref_name))


def get_user_payments_count(user_id: int) -> int:
    ensure_payments_table()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(1) FROM payments WHERE user_id=%s", (user_id,))
            row = cur.fetchone()
            return int(row[0] or 0)


def apply_affiliate_commission_for_first_purchase(user_id: int, charge_id: str, amount: int) -> None:
    """If user has referrer and this is the first payment, accrue commission to partner balance and store referral row."""
    # Check first purchase
    cnt = get_user_payments_count(user_id)
    if cnt != 1:
        return
    # Load referrer_code from users
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT referrer_code FROM users WHERE user_id=%s",
                (user_id,),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return
            ref_code = str(row[0])
    aff = get_affiliate_by_code(ref_code)
    if not aff:
        return
    partner_user_id, percent, partner_name = aff
    commission = int(round(amount * percent / 100.0))
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Accrue balance
            cur.execute(
                "UPDATE affiliates SET balance=balance+%s WHERE partner_user_id=%s",
                (commission, partner_user_id),
            )
            # Insert referral row
            cur.execute(
                "INSERT INTO referrals (partner_user_id, referred_user_id, code, charge_id, amount, commission) VALUES (%s, %s, %s, %s, %s, %s)",
                (partner_user_id, user_id, ref_code, charge_id, amount, commission),
            )


def get_affiliate_stats(partner_user_id: int) -> tuple[int, int]:
    ensure_affiliates_tables()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(1), COALESCE(SUM(commission),0) FROM referrals WHERE partner_user_id=%s",
                (partner_user_id,),
            )
            row = cur.fetchone() or (0, 0)
            return int(row[0] or 0), int(row[1] or 0)


def get_user_referrer_info(user_id: int) -> tuple[str | None, int | None, int | None, str | None]:
    """Return (ref_code, partner_user_id, percent, partner_name) for a user, if any."""
    ensure_affiliates_tables()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT referrer_code FROM users WHERE user_id=%s", (user_id,))
            row = cur.fetchone()
            if not row or not row[0]:
                return None, None, None, None
            code = str(row[0])
            cur.execute(
                "SELECT partner_user_id, percent, partner_name FROM affiliates WHERE code=%s",
                (code,),
            )
            row2 = cur.fetchone()
            if not row2:
                return code, None, None, None
            return code, int(row2[0]), int(row2[1]), (row2[2] if row2[2] is not None else None)


def list_referrals(partner_user_id: int, limit: int = 10) -> list[tuple[int, str, int, int, str]]:
    """Return recent referrals for a partner: (referred_user_id, charge_id, amount, commission, created_at_iso)."""
    ensure_affiliates_tables()
    out: list[tuple[int, str, int, int, str]] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT referred_user_id, charge_id, amount, commission, created_at FROM referrals WHERE partner_user_id=%s ORDER BY created_at DESC LIMIT %s",
                (partner_user_id, limit),
            )
            rows = cur.fetchall() or []
            for r in rows:
                out.append((int(r[0]), str(r[1] or ""), int(r[2] or 0), int(r[3] or 0), (r[4].isoformat() if r[4] else "")))
    return out


# ============ Affiliate requests ============

def ensure_affiliate_requests_table() -> None:
    sql = (
        "CREATE TABLE IF NOT EXISTS affiliate_requests (\n"
        "  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),\n"
        "  user_id BIGINT NOT NULL,\n"
        "  username TEXT,\n"
        "  note TEXT,\n"
        "  status TEXT NOT NULL DEFAULT 'pending',\n"
        "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),\n"
        "  processed_at TIMESTAMPTZ\n"
        ")"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def insert_affiliate_request(user_id: int, username: str | None, note: str | None = None) -> str:
    ensure_affiliate_requests_table()
    sql = "INSERT INTO affiliate_requests (user_id, username, note) VALUES (%s, %s, %s) RETURNING id"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, username, note))
            row = cur.fetchone()
            return str(row[0])


def list_affiliate_requests(status: str = "pending", limit: int = 20) -> list[tuple[str, int, str | None, str | None, str]]:
    ensure_affiliate_requests_table()
    sql = (
        "SELECT id, user_id, username, note, created_at FROM affiliate_requests "
        "WHERE status=%s ORDER BY created_at DESC LIMIT %s"
    )
    out: list[tuple[str, int, str | None, str | None, str]] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (status, limit))
            rows = cur.fetchall() or []
            for r in rows:
                out.append((str(r[0]), int(r[1]), (r[2] if r[2] is not None else None), (r[3] if r[3] is not None else None), r[4].isoformat() if r[4] else ""))
    return out


def mark_affiliate_request(id_str: str, status: str) -> None:
    ensure_affiliate_requests_table()
    sql = "UPDATE affiliate_requests SET status=%s, processed_at=now() WHERE id=%s"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (status, id_str))



def insert_news_signals(rows: Iterable[dict]) -> int:
    sql = (
        "INSERT INTO news_signals (src_id, ts, title, source, url, sentiment, topics, impact_score, confidence) "
        "VALUES (%(src_id)s, %(ts)s, %(title)s, %(source)s, %(url)s, %(sentiment)s, %(topics)s, %(impact_score)s, %(confidence)s) "
        "ON CONFLICT (src_id) DO NOTHING"
    )
    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(sql, r)
                count += cur.rowcount
    logger.info(f"Inserted news rows: {count}")
    return count


def insert_news_facts(rows: Iterable[dict]) -> int:
    """Insert LLM-extracted news facts (best-effort).

    Expected keys per row:
      - src_id: str
      - ts: datetime or iso str
      - type: str
      - direction: str
      - magnitude: float
      - confidence: float
      - entities: dict/list (JSON serializable)
      - raw: dict (original extraction payload)
    """
    sql = (
        "INSERT INTO news_facts (src_id, ts, type, direction, magnitude, confidence, entities, raw) "
        "VALUES (%(src_id)s, %(ts)s, %(type)s, %(direction)s, %(magnitude)s, %(confidence)s, %(entities)s, %(raw)s) "
        "ON CONFLICT (src_id, type, ts) DO NOTHING"
    )
    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(
                    sql,
                    {
                        "src_id": r.get("src_id"),
                        "ts": r.get("ts"),
                        "type": r.get("type"),
                        "direction": r.get("direction"),
                        "magnitude": r.get("magnitude"),
                        "confidence": r.get("confidence"),
                        "entities": Json(r.get("entities") or {}),
                        "raw": Json(r.get("raw") or {}),
                    },
                )
                count += cur.rowcount
    logger.info(f"Inserted news facts: {count}")
    return count


def fetch_news_facts_by_type(typ: str, limit: int = 50) -> list[tuple[str, str | None, float | None, float | None, str | None]]:
    """Return recent news facts of a given type.

    Output: list of (ts_iso, direction, magnitude, confidence, src_id)
    """
    sql = (
        "SELECT ts, direction, magnitude, confidence, src_id FROM news_facts "
        "WHERE type=%s ORDER BY ts DESC LIMIT %s"
    )
    out: list[tuple[str, str | None, float | None, float | None, str | None]] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (typ, int(limit)))
            rows = cur.fetchall() or []
            for r in rows:
                ts, direction, mag, conf, src_id = r
                out.append(
                    (
                        (ts.isoformat() if hasattr(ts, "isoformat") else str(ts)),
                        (str(direction) if direction is not None else None),
                        (float(mag) if mag is not None else None),
                        (float(conf) if conf is not None else None),
                        (str(src_id) if src_id is not None else None),
                    )
                )
    return out


def upsert_prediction(
    run_id: str,
    horizon: str,
    y_hat: float,
    pi_low: float,
    pi_high: float,
    proba_up: float,
    per_model_json: dict,
) -> None:
    sql = (
        "INSERT INTO predictions (run_id, horizon, y_hat, pi_low, pi_high, proba_up, per_model_json) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (run_id, horizon) DO UPDATE SET y_hat=excluded.y_hat, pi_low=excluded.pi_low, pi_high=excluded.pi_high, proba_up=excluded.proba_up, per_model_json=excluded.per_model_json"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    run_id,
                    horizon,
                    y_hat,
                    pi_low,
                    pi_high,
                    proba_up,
                    Json(per_model_json),
                ),
            )
    logger.info(f"Saved prediction {run_id} {horizon}")


def upsert_prediction_outcome(
    run_id: str,
    horizon: str,
    created_at_iso: str,
    y_hat: float | None,
    y_true: float | None,
    error_abs: float | None,
    error_pct: float | None,
    direction_correct: bool | None,
    regime_label: str | None,
    news_ctx: float | None,
    tags: dict | None = None,
) -> None:
    """Store realized outcome for a prediction run (id + horizon primary key)."""
    sql = (
        "INSERT INTO prediction_outcomes (run_id, horizon, created_at, y_hat, y_true, error_abs, error_pct, direction_correct, regime_label, news_ctx, tags) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (run_id, horizon) DO UPDATE SET created_at=EXCLUDED.created_at, y_hat=EXCLUDED.y_hat, y_true=EXCLUDED.y_true, error_abs=EXCLUDED.error_abs, error_pct=EXCLUDED.error_pct, direction_correct=EXCLUDED.direction_correct, regime_label=EXCLUDED.regime_label, news_ctx=EXCLUDED.news_ctx, tags=EXCLUDED.tags"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    run_id,
                    horizon,
                    created_at_iso,
                    y_hat,
                    y_true,
                    error_abs,
                    error_pct,
                    direction_correct,
                    regime_label,
                    news_ctx,
                    Json(tags or {}),
                ),
            )
    logger.info(f"Saved prediction outcome {run_id} {horizon}")


def upsert_features_snapshot(run_id: str, ts_window_iso: str, path_s3: str) -> None:
    sql = (
        "INSERT INTO features_snapshot (run_id, ts_window, path_s3) VALUES (%s, %s, %s) "
        "ON CONFLICT (run_id) DO UPDATE SET ts_window=excluded.ts_window, path_s3=excluded.path_s3"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id, ts_window_iso, path_s3))
    logger.info(f"Saved features snapshot for {run_id}")


def upsert_ensemble_weights(run_id: str, weights: dict) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            for model, w in weights.items():
                cur.execute(
                    "INSERT INTO ensemble_weights (run_id, model, w) VALUES (%s, %s, %s) "
                    "ON CONFLICT (run_id, model) DO UPDATE SET w=excluded.w",
                    (run_id, model, w),
                )
    logger.info(f"Saved ensemble weights for {run_id}")


def upsert_model_trust_regime(
    regime_label: str, horizon: str, model: str, weight: float
) -> None:
    """Upsert per-regime model trust weight."""
    sql = (
        "INSERT INTO model_trust_regime (regime_label, horizon, model, weight) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (regime_label, horizon, model) DO UPDATE SET weight=EXCLUDED.weight, updated_at=now()"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (regime_label, horizon, model, weight))


def fetch_model_trust_regime(regime_label: str, horizon: str) -> dict:
    """Fetch model->weight map for a given regime and horizon.

    Returns empty dict if none found.
    """
    sql = "SELECT model, weight FROM model_trust_regime WHERE regime_label=%s AND horizon=%s"
    out: dict[str, float] = {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (regime_label, horizon))
            rows = cur.fetchall() or []
            for m, w in rows:
                out[str(m)] = float(w or 0.0)
    return out


def upsert_model_trust_regime_event(
    regime_label: str, horizon: str, event_type: str, model: str, weight: float
) -> None:
    sql = (
        "INSERT INTO model_trust_regime_event (regime_label, horizon, event_type, model, weight) VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (regime_label, horizon, event_type, model) DO UPDATE SET weight=EXCLUDED.weight, updated_at=now()"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (regime_label, horizon, event_type, model, weight))


def fetch_model_trust_regime_event(regime_label: str, horizon: str, event_type: str) -> dict:
    sql = (
        "SELECT model, weight FROM model_trust_regime_event WHERE regime_label=%s AND horizon=%s AND event_type=%s"
    )
    out: dict[str, float] = {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (regime_label, horizon, event_type))
            rows = cur.fetchall() or []
            for m, w in rows:
                out[str(m)] = float(w or 0.0)
    return out


def upsert_scenarios(run_id: str, scenarios_list: list, charts_s3: str | None) -> None:
    sql = (
        "INSERT INTO scenarios (run_id, list_json, charts_s3) VALUES (%s, %s, %s) "
        "ON CONFLICT (run_id) DO UPDATE SET list_json=excluded.list_json, charts_s3=excluded.charts_s3"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id, Json(scenarios_list), charts_s3))
    logger.info(f"Saved scenarios for {run_id}")


def upsert_trade_suggestion(run_id: str, card: dict) -> None:
    sql = (
        "INSERT INTO trades_suggestions (run_id, side, entry_zone, leverage, sl, tp, rr, reason_codes, times_json) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (run_id) DO UPDATE SET side=excluded.side, entry_zone=excluded.entry_zone, leverage=excluded.leverage, sl=excluded.sl, tp=excluded.tp, rr=excluded.rr, reason_codes=excluded.reason_codes, times_json=excluded.times_json"
    )
    entry_zone = card.get("entry", {}).get("zone")
    times = {
        "signal_time": card.get("signal_time"),
        "valid_until": card.get("valid_until"),
        "horizon_ends_at": card.get("horizon_ends_at"),
    }
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    run_id,
                    card.get("side"),
                    Json(entry_zone),
                    card.get("leverage"),
                    card.get("stop_loss"),
                    card.get("take_profit"),
                    card.get("rr_expected"),
                    Json(card.get("reason_codes")),
                    Json(times),
                ),
            )
    logger.info(f"Saved trade suggestion for {run_id}")


def upsert_regime(
    run_id: str, label: str, confidence: float, regime_features: dict
) -> None:
    sql = (
        "INSERT INTO regimes (run_id, label, confidence) VALUES (%s, %s, %s) "
        "ON CONFLICT (run_id) DO UPDATE SET label=excluded.label, confidence=excluded.confidence"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id, label, confidence))
    logger.info(f"Saved regime for {run_id}: {label} ({confidence})")


def log_error(
    run_id: str,
    horizon: str | None,
    metrics_json: dict | None,
    regime: str | None,
    features_digest: str | None,
) -> None:
    sql = "INSERT INTO errors_log (run_id, horizon, metrics_json, regime, features_digest) VALUES (%s, %s, %s, %s, %s)"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (run_id, horizon, Json(metrics_json or {}), regime, features_digest),
            )


def upsert_explanations(run_id: str, markdown: str, risk_flags: list[str]) -> None:
    sql = (
        "INSERT INTO explanations (run_id, markdown, risk_flags) VALUES (%s, %s, %s) "
        "ON CONFLICT (run_id) DO UPDATE SET markdown=excluded.markdown, risk_flags=excluded.risk_flags"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id, markdown, Json(risk_flags)))
    logger.info(f"Saved explanations for {run_id}")


def insert_social_signals(rows: Iterable[dict]) -> int:
    sql = (
        "INSERT INTO social_signals (src_id, ts, platform, author, text, url, sentiment, score, topics, metrics) "
        "VALUES (%(src_id)s, %(ts)s, %(platform)s, %(author)s, %(text)s, %(url)s, %(sentiment)s, %(score)s, %(topics)s, %(metrics)s) "
        "ON CONFLICT (src_id) DO NOTHING"
    )
    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(sql, r)
                count += cur.rowcount
    logger.info(f"Inserted social rows: {count}")
    return count


def insert_onchain_signals(rows: Iterable[dict]) -> int:
    sql = (
        "INSERT INTO onchain_signals (run_id, ts, asset, metric, value, interpretation) "
        "VALUES (%(run_id)s, %(ts)s, %(asset)s, %(metric)s, %(value)s, %(interpretation)s)"
    )
    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(sql, r)
                count += cur.rowcount
    logger.info(f"Inserted onchain rows: {count}")
    return count


def insert_futures_metrics(rows: Iterable[dict]) -> int:
    sql = (
        "INSERT INTO futures_metrics (run_id, ts, exchange, symbol, funding_rate, next_funding_time, mark_price, index_price, open_interest) "
        "VALUES (%(run_id)s, %(ts)s, %(exchange)s, %(symbol)s, %(funding_rate)s, %(next_funding_time)s, %(mark_price)s, %(index_price)s, %(open_interest)s)"
    )
    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(sql, r)
                count += cur.rowcount
    logger.info(f"Inserted futures rows: {count}")
    return count


def upsert_social_embedding(
    src_id: str, platform: str, ts: str, embedding: List[float], meta: dict
) -> None:
    ensure_vector()
    sql = (
        "INSERT INTO social_index (src_id, platform, ts, dim, embedding, meta) VALUES (%s, %s, %s, %s, %s::vector, %s) "
        "ON CONFLICT (src_id) DO UPDATE SET platform=excluded.platform, ts=excluded.ts, dim=excluded.dim, embedding=excluded.embedding, meta=excluded.meta"
    )
    vec = "[" + ",".join(f"{float(v):.6f}" for v in embedding) + "]"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (src_id, platform, ts, len(embedding), vec, Json(meta)))


def query_social_similar(embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
    ensure_vector()
    sql = (
        "SELECT src_id::text, (embedding <-> %s::vector) AS distance FROM social_index "
        "ORDER BY embedding <-> %s::vector LIMIT %s"
    )
    vec = "[" + ",".join(f"{float(v):.6f}" for v in embedding) + "]"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (vec, vec, k))
            rows = cur.fetchall() or []
    return [(str(r[0]), float(r[1])) for r in rows]


def insert_alt_signals(rows: Iterable[dict]) -> int:
    sql = (
        "INSERT INTO alt_signals (run_id, ts, source, metric, value, meta) "
        "VALUES (%(run_id)s, %(ts)s, %(source)s, %(metric)s, %(value)s, %(meta)s)"
    )
    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(sql, r)
                count += cur.rowcount
    logger.info(f"Inserted alt rows: {count}")
    return count


def ensure_vector() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")


def upsert_feature_embedding(
    ts_window: str, symbol: str, embedding: List[float], meta: dict
) -> None:
    ensure_vector()
    sql = (
        "INSERT INTO features_index (ts_window, symbol, dim, embedding, meta) VALUES (%s, %s, %s, %s::vector, %s) "
        "ON CONFLICT (ts_window) DO UPDATE SET symbol=excluded.symbol, dim=excluded.dim, embedding=excluded.embedding, meta=excluded.meta"
    )
    vec = "[" + ",".join(f"{float(v):.6f}" for v in embedding) + "]"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ts_window, symbol, len(embedding), vec, Json(meta)))


def query_similar(embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
    ensure_vector()
    sql = (
        "SELECT ts_window::text, (embedding <-> %s::vector) AS distance FROM features_index "
        "ORDER BY embedding <-> %s::vector LIMIT %s"
    )
    vec = "[" + ",".join(f"{float(v):.6f}" for v in embedding) + "]"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (vec, vec, k))
            rows = cur.fetchall() or []
    return [(r[0], float(r[1])) for r in rows]


def upsert_similar_windows(run_id: str, topk_json: list) -> None:
    sql = (
        "INSERT INTO similar_windows (run_id, topk_json) VALUES (%s, %s) "
        "ON CONFLICT (run_id) DO UPDATE SET topk_json=excluded.topk_json"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id, Json(topk_json)))


def fetch_recent_predictions(
    limit: int = 5, horizon: str = "4h"
) -> list[tuple[str, str, float, float, str]]:
    """Fetch recent predictions for memory context.

    Returns list of (run_id, horizon, y_hat, proba_up, created_at_iso).
    """
    sql = "SELECT run_id, horizon, y_hat, proba_up, created_at FROM predictions WHERE horizon=%s ORDER BY created_at DESC LIMIT %s"
    out: list[tuple[str, str, float, float, str]] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (horizon, limit))
            for r in cur.fetchall() or []:
                rid, hz, y, p, ts = r
                out.append(
                    (
                        str(rid),
                        str(hz),
                        float(y or 0.0),
                        float(p or 0.0),
                        ts.isoformat() if ts else "",
                    )
                )
    return out


def upsert_agent_prediction(agent: str, run_id: str, result_json: dict) -> None:
    sql = (
        "INSERT INTO agents_predictions (run_id, agent, result_json) VALUES (%s, %s, %s) "
        "ON CONFLICT (run_id, agent) DO UPDATE SET result_json=excluded.result_json"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id, agent, Json(result_json)))


def insert_agent_metric(
    agent: str, metric: str, value: float, labels: dict | None = None, ts=None
) -> None:
    sql = "INSERT INTO agents_metrics (ts, agent, metric, value, labels) VALUES (%s, %s, %s, %s, %s)"
    from datetime import datetime
    from datetime import timezone as _tz

    ts = ts or datetime.now(_tz.utc)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ts, agent, metric, value, Json(labels or {})))


def upsert_validation_report(
    run_id: str,
    status: str,
    items: dict,
    warnings: list | None = None,
    errors: list | None = None,
) -> None:
    sql = (
        "INSERT INTO validation_reports (run_id, status, items, warnings, errors) VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (run_id) DO UPDATE SET status=excluded.status, items=excluded.items, warnings=excluded.warnings, errors=excluded.errors"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    run_id,
                    status,
                    Json(items or {}),
                    Json(warnings or []),
                    Json(errors or []),
                ),
            )


def insert_backtest_result(cfg: dict, metrics: dict) -> None:
    sql = "INSERT INTO backtest_results (cfg_json, metrics_json) VALUES (%s, %s)"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (Json(cfg or {}), Json(metrics or {})))


def upsert_model_registry(
    name: str,
    version: str,
    path_s3: str | None = None,
    params: dict | None = None,
    metrics: dict | None = None,
) -> None:
    """Insert or update a model entry in model_registry table.

    Requires migration 012_model_registry.sql. Best-effort: swallows exceptions to not break pipeline.
    """
    sql = (
        "INSERT INTO model_registry (name, version, path_s3, params, metrics) VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (name, version) DO UPDATE SET path_s3=excluded.path_s3, params=excluded.params, metrics=excluded.metrics"
    )
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (name, version, path_s3, Json(params or {}), Json(metrics or {})),
                )
    except Exception:
        logger.warning("Failed to upsert model_registry", exc_info=True)


def fetch_predictions_for_cv(
    horizon: str, min_age_hours: float = 1.0
) -> list[tuple[str, str, str, float, float, float, dict]]:
    """Return matured predictions: (run_id, horizon, created_at_iso, y_hat, pi_low, pi_high, per_model_json)."""
    sql = "SELECT run_id, horizon, created_at, y_hat, pi_low, pi_high, per_model_json FROM predictions WHERE horizon=%s AND created_at <= now() - (%s || ' hours')::interval ORDER BY created_at DESC LIMIT 200"
    out: list[tuple[str, str, str, float, float, float, dict]] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (horizon, str(float(min_age_hours))))
            rows = cur.fetchall() or []
            for r in rows:
                run_id, hz, created_at, y_hat, pi_low, pi_high, per_model = r
                out.append(
                    (
                        str(run_id),
                        str(hz),
                        created_at.isoformat() if created_at else None,
                        float(y_hat or 0.0),
                        float(pi_low or 0.0),
                        float(pi_high or 0.0),
                        per_model or {},
                    )
                )
    return out


def fetch_agent_metrics(
    agent: str, metric_like: str, limit: int = 5
) -> list[tuple[str, float]]:
    """Fetch recent agents_metrics by agent and metric prefix.

    Returns list of (ts_iso, value).
    """
    sql = "SELECT ts, value FROM agents_metrics WHERE agent=%s AND metric LIKE %s ORDER BY ts DESC LIMIT %s"
    out: list[tuple[str, float]] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (agent, metric_like, limit))
            rows = cur.fetchall() or []
            for ts, v in rows:
                out.append((ts.isoformat() if ts else "", float(v or 0.0)))
    return out


def ensure_subscriptions_tables() -> None:
    sql = (
        "CREATE TABLE IF NOT EXISTS subscriptions (\n"
        "  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),\n"
        "  user_id BIGINT NOT NULL,\n"
        "  provider TEXT NOT NULL,\n"
        "  status TEXT NOT NULL,\n"
        "  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),\n"
        "  ends_at TIMESTAMPTZ NOT NULL,\n"
        "  payload JSONB,\n"
        "  UNIQUE (user_id, status)\n"
        ")"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def ensure_payments_table() -> None:
    sql = (
        "CREATE TABLE IF NOT EXISTS payments (\n"
        "  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),\n"
        "  user_id BIGINT NOT NULL,\n"
        "  charge_id TEXT UNIQUE NOT NULL,\n"
        "  amount BIGINT NOT NULL,\n"
        "  status TEXT NOT NULL DEFAULT 'paid',\n"
        "  payload JSONB,\n"
        "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n"
        ")"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def payment_exists(charge_id: str) -> bool:
    ensure_payments_table()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM payments WHERE charge_id=%s", (charge_id,))
            return cur.fetchone() is not None


def insert_payment(charge_id: str, user_id: int, amount: int) -> None:
    ensure_payments_table()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO payments (charge_id, user_id, amount) VALUES (%s, %s, %s)",
                (charge_id, user_id, amount),
            )


def add_subscription(user_id: int, provider: str, months: int, payload: dict) -> None:
    ensure_subscriptions_tables()
    now = datetime.now(timezone.utc)
    ends = now + timedelta(days=30 * months)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO subscriptions (user_id, provider, status, started_at, ends_at, payload) VALUES (%s, %s, 'active', %s, %s, %s) "
                "ON CONFLICT (user_id, status) DO UPDATE SET ends_at=GREATEST(subscriptions.ends_at, EXCLUDED.ends_at), payload=EXCLUDED.payload",
                (user_id, provider, now, ends, Json(payload)),
            )


def create_redeem_code(months: int, invoice_id: str) -> str:
    """Generate and store a one-time redeem code for a given invoice."""
    import secrets

    code = secrets.token_urlsafe(8)
    sql = "INSERT INTO redeem_codes (code, months, invoice_id) VALUES (%s, %s, %s)"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (code, months, invoice_id))
    return code


def redeem_code_use(code: str) -> int | None:
    """Mark redeem code as used and return months granted."""
    sql = (
        "UPDATE redeem_codes SET used_at=now() WHERE code=%s AND used_at IS NULL RETURNING months"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (code,))
            row = cur.fetchone()
            if not row:
                return None
            return int(row[0])


def get_subscription_status(user_id: int) -> tuple[str, Optional[str]]:
    ensure_subscriptions_tables()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, ends_at FROM subscriptions WHERE user_id=%s ORDER BY ends_at DESC LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return "none", None
            status, ends_at = row
            if ends_at and ends_at > datetime.now(timezone.utc):
                return str(status), ends_at.isoformat()
            return "expired", ends_at.isoformat() if ends_at else None


def sweep_expired_subscriptions(now=None) -> int:
    ensure_subscriptions_tables()
    from datetime import datetime, timezone

    now = now or datetime.now(timezone.utc)
    sql = "UPDATE subscriptions SET status='expired' WHERE status='active' AND ends_at <= %s RETURNING user_id"
    count = 0
    users = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (now,))
            rows = cur.fetchall() or []
            users = [int(r[0]) for r in rows]
            count = len(users)
    return count


    # duplicate older definition removed; unified above


def mark_payment_refunded(charge_id: str) -> int | None:
    """Mark payment refunded and return user_id if found."""
    ensure_payments_table()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE payments SET status='refunded' WHERE charge_id=%s RETURNING user_id",
                (charge_id,),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None


def mark_subscription_refunded(user_id: int) -> None:
    ensure_subscriptions_tables()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE subscriptions SET status='refunded' WHERE user_id=%s",
                (user_id,),
            )


def insert_user_feedback(
    user_id: int,
    run_id: str,
    rating: int | None,
    comment: str | None,
    meta: dict | None = None,
) -> None:
    sql = "INSERT INTO users_feedback (user_id, run_id, rating, comment, meta) VALUES (%s, %s, %s, %s, %s)"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, run_id, rating, comment, Json(meta or {})))


# --- User insights (community signals) -------------------------------------------------

def ensure_user_insights_table() -> None:
    sql = (
        "CREATE TABLE IF NOT EXISTS user_insights (\n"
        "  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),\n"
        "  user_id BIGINT NOT NULL,\n"
        "  text TEXT NOT NULL,\n"
        "  url TEXT,\n"
        "  verdict TEXT,\n"
        "  score_truth NUMERIC,\n"
        "  score_freshness NUMERIC,\n"
        "  meta JSONB,\n"
        "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n"
        ")"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def insert_user_insight(
    user_id: int,
    text: str,
    url: Optional[str] = None,
    verdict: Optional[str] = None,
    score_truth: Optional[float] = None,
    score_freshness: Optional[float] = None,
    meta: Optional[dict] = None,
) -> str:
    ensure_user_insights_table()
    sql = (
        "INSERT INTO user_insights (user_id, text, url, verdict, score_truth, score_freshness, meta) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    user_id,
                    text,
                    url,
                    verdict,
                    score_truth,
                    score_freshness,
                    Json(meta or {}),
                ),
            )
            row = cur.fetchone()
            return str(row[0])


def fetch_user_insights_recent(
    hours: int = 24,
    min_truth: float = 0.6,
    min_freshness: float = 0.5,
    limit: int = 50,
) -> list[dict]:
    ensure_user_insights_table()
    sql = (
        "SELECT user_id, text, url, verdict, score_truth, score_freshness, created_at "
        "FROM user_insights "
        "WHERE created_at >= now() - (%s || ' hours')::interval "
        "AND COALESCE(score_truth, 0) >= %s AND COALESCE(score_freshness, 0) >= %s "
        "ORDER BY created_at DESC LIMIT %s"
    )
    out: list[dict] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (str(float(hours)), float(min_truth), float(min_freshness), int(limit)))
            rows = cur.fetchall() or []
            for r in rows:
                uid, text, url, verdict, st, sf, created_at = r
                out.append(
                    {
                        "user_id": int(uid),
                        "text": str(text or ""),
                        "url": str(url or "") or None,
                        "verdict": str(verdict or ""),
                        "score_truth": float(st or 0.0),
                        "score_freshness": float(sf or 0.0),
                        "created_at": created_at,
                    }
                )
    return out


# --- Run summaries (agent memory) -------------------------------------------

def insert_run_summary(
    run_id: str,
    final_analysis: dict | None = None,
    prediction_outcome: dict | None = None,
) -> None:
    """Insert or update compact summary for a run used as agent memory."""
    sql = (
        "INSERT INTO run_summaries (run_id, final_analysis_json, prediction_outcome_json) "
        "VALUES (%s, %s, %s) "
        "ON CONFLICT (run_id) DO UPDATE SET final_analysis_json=EXCLUDED.final_analysis_json, prediction_outcome_json=EXCLUDED.prediction_outcome_json"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id, Json(final_analysis or {}), Json(prediction_outcome or {})))


def fetch_recent_run_summaries(n: int = 3) -> list[dict]:
    """Return recent run summaries as list of dicts with keys: run_id, created_at, final, outcome."""
    sql = (
        "SELECT run_id, created_at, final_analysis_json, prediction_outcome_json FROM run_summaries "
        "ORDER BY created_at DESC LIMIT %s"
    )
    out: list[dict] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (int(n),))
            rows = cur.fetchall() or []
            for r in rows:
                run_id, created_at, final, outcome = r
                out.append(
                    {
                        "run_id": str(run_id),
                        "created_at": created_at.isoformat() if created_at else "",
                        "final": final or {},
                        "outcome": outcome or {},
                    }
                )
    return out


# --- Technical patterns knowledge base -------------------------------------

def fetch_technical_patterns() -> list[dict]:
    """Return list of technical pattern definitions from DB.

    Each item has keys: name, category, timeframe, definition, description, source, confidence_default.
    """
    sql = (
        "SELECT name, category, timeframe, definition_json, description, source, confidence_default "
        "FROM technical_patterns ORDER BY name"
    )
    out: list[dict] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql)
                rows = cur.fetchall() or []
            except Exception:
                rows = []
            for r in rows:
                name, category, timeframe, definition, description, source, conf = r
                out.append(
                    {
                        "name": str(name),
                        "category": str(category or ""),
                        "timeframe": str(timeframe or ""),
                        "definition": definition or {},
                        "description": str(description or ""),
                        "source": str(source or ""),
                        "confidence_default": float(conf or 0.0),
                    }
                )
    return out


def upsert_technical_pattern(
    name: str,
    category: str,
    timeframe: str,
    definition: dict,
    description: str | None = None,
    source: str | None = None,
    confidence_default: float = 0.6,
) -> None:
    sql = (
        "INSERT INTO technical_patterns (name, category, timeframe, definition_json, description, source, confidence_default) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (name) DO UPDATE SET category=EXCLUDED.category, timeframe=EXCLUDED.timeframe, definition_json=EXCLUDED.definition_json, description=EXCLUDED.description, source=EXCLUDED.source, confidence_default=EXCLUDED.confidence_default"
    )
    from psycopg2.extras import Json as _Json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (name, category, timeframe, _Json(definition or {}), description, source, float(confidence_default)))


def fetch_agent_config(agent_name: str) -> dict | None:
    sql = (
        "SELECT system_prompt, parameters_json FROM agent_configurations "
        "WHERE agent_name=%s AND is_active=true ORDER BY version DESC LIMIT 1"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (agent_name,))
            row = cur.fetchone()
            if not row:
                return None
            sys, params = row
            return {"system_prompt": sys or "", "parameters": params or {}}


def insert_agent_config(
    agent_name: str,
    system_prompt: str | None,
    parameters: dict | None,
    make_active: bool = True,
) -> int:
    """Insert a new version for an agent; returns version number.

    If make_active=True, deactivate previous active versions for this agent.
    """
    # fetch latest version
    sel = "SELECT COALESCE(MAX(version), 0) FROM agent_configurations WHERE agent_name=%s"
    upd_off = "UPDATE agent_configurations SET is_active=false WHERE agent_name=%s AND is_active=true"
    ins = (
        "INSERT INTO agent_configurations (agent_name, version, system_prompt, parameters_json, is_active) "
        "VALUES (%s, %s, %s, %s, %s)"
    )
    from psycopg2.extras import Json as _Json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sel, (agent_name,))
            ver = int((cur.fetchone() or [0])[0]) + 1
            if make_active:
                cur.execute(upd_off, (agent_name,))
            cur.execute(ins, (agent_name, ver, system_prompt, _Json(parameters or {}), bool(make_active)))
            return ver


def insert_pattern_metrics(
    symbol: str,
    timeframe: str,
    window_hours: int,
    move_threshold: float,
    sample_count: int,
    pattern_name: str,
    expected_direction: str,
    match_count: int,
    success_count: int,
    success_rate: float,
    p_value: float,
    definition: dict | None = None,
    summary: dict | None = None,
) -> None:
    sql = (
        "INSERT INTO pattern_discovery_metrics "
        "(symbol, timeframe, window_hours, move_threshold, sample_count, pattern_name, expected_direction, match_count, success_count, success_rate, p_value, definition_json, summary_json) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    )
    from psycopg2.extras import Json as _Json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    symbol,
                    timeframe,
                    int(window_hours),
                    float(move_threshold),
                    int(sample_count),
                    pattern_name,
                    expected_direction,
                    int(match_count),
                    int(success_count),
                    float(success_rate),
                    float(p_value),
                    _Json(definition or {}),
                    _Json(summary or {}),
                ),
            )


# --- Lessons (compressed memory) -------------------------------------------

def insert_agent_lesson(lesson_text: str, scope: str = "global", meta: dict | None = None) -> None:
    sql = "INSERT INTO agent_lessons (lesson_text, scope, meta) VALUES (%s, %s, %s)"
    from psycopg2.extras import Json as _Json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (lesson_text, scope, _Json(meta or {})))


def fetch_recent_agent_lessons(n: int = 5) -> list[dict]:
    sql = "SELECT created_at, scope, lesson_text, meta FROM agent_lessons ORDER BY created_at DESC LIMIT %s"
    out: list[dict] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (int(n),))
            rows = cur.fetchall() or []
            for created_at, scope, txt, meta in rows:
                out.append(
                    {
                        "created_at": created_at.isoformat() if created_at else "",
                        "scope": str(scope or "global"),
                        "lesson_text": str(txt or ""),
                        "meta": meta or {},
                    }
                )
    return out
