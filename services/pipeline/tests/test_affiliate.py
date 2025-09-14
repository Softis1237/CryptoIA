import os
import sqlite3
import sys
from contextlib import contextmanager

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
from pipeline.infra import db  # noqa: E402


class SQLiteCursor:
    def __init__(self, cursor):
        self.cur = cursor

    def execute(self, sql, params=None):
        if params is not None:
            sql = sql.replace("%s", "?")
            self.cur.execute(sql, params)
        else:
            self.cur.execute(sql)
        return self

    def fetchone(self):
        return self.cur.fetchone()

    def fetchall(self):
        return self.cur.fetchall()

    # support context manager protocol
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cur.close()


class SQLiteConn:
    def __init__(self, conn):
        self.conn = conn

    def cursor(self):
        return SQLiteCursor(self.conn.cursor())

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        pass


@pytest.fixture
def db_patch(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE affiliates (
            partner_user_id INTEGER PRIMARY KEY,
            partner_name TEXT,
            code TEXT UNIQUE,
            percent INT,
            balance INT DEFAULT 0
        );
        CREATE TABLE referrals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            partner_user_id INTEGER,
            referred_user_id INTEGER,
            code TEXT,
            charge_id TEXT,
            amount INT,
            commission INT
        );
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            referrer_code TEXT,
            referrer_name TEXT
        );
        CREATE TABLE payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            charge_id TEXT UNIQUE,
            amount INT,
            status TEXT,
            payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    wrapper = SQLiteConn(conn)

    @contextmanager
    def _get_conn():
        try:
            yield wrapper
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    monkeypatch.setattr(db, "get_conn", lambda: _get_conn())
    monkeypatch.setattr(db, "ensure_affiliates_tables", lambda: None)
    monkeypatch.setattr(db, "ensure_payments_table", lambda: None)
    return conn


def test_get_or_create_affiliate(db_patch):
    code, pct = db.get_or_create_affiliate(1, "Alice", 60)
    assert code == "A1"
    assert pct == 60
    code2, pct2 = db.get_or_create_affiliate(1, "Alice", 60)
    assert code2 == code
    assert pct2 == pct


def test_apply_affiliate_commission_for_first_purchase(db_patch):
    # partner creates affiliate code
    code, pct = db.get_or_create_affiliate(10, "Bob", 50)
    conn = db_patch
    # user has referrer code
    conn.execute(
        "INSERT INTO users (user_id, referrer_code) VALUES (?, ?)",
        (20, code),
    )
    # first payment
    conn.execute(
        "INSERT INTO payments (charge_id, user_id, amount) VALUES (?, ?, ?)",
        ("c1", 20, 1000),
    )
    db.apply_affiliate_commission_for_first_purchase(20, "c1", 1000)
    cur = conn.execute(
        "SELECT balance FROM affiliates WHERE partner_user_id=?",
        (10,),
    )
    assert cur.fetchone()[0] == 500
    cur = conn.execute(
        (
            "SELECT partner_user_id, referred_user_id, amount, commission "
            "FROM referrals"
        )
    )
    assert cur.fetchone() == (10, 20, 1000, 500)
    # second payment should not accrue
    conn.execute(
        "INSERT INTO payments (charge_id, user_id, amount) VALUES (?, ?, ?)",
        ("c2", 20, 2000),
    )
    db.apply_affiliate_commission_for_first_purchase(20, "c2", 2000)
    cur = conn.execute("SELECT COUNT(1) FROM referrals")
    assert cur.fetchone()[0] == 1
