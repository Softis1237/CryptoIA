import os
import hmac
import hashlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# Ensure package path is importable when tests are run from repo root
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_pay_api.main import app


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Clear secret by default; tests set explicitly where needed
    monkeypatch.delenv("CRYPTO_PAY_SECRET", raising=False)
    yield


def _sign(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_confirm_requires_secret():
    client = TestClient(app)
    payload = {"invoice_id": "inv1", "plan": 1, "telegram_id": 123}
    body = json.dumps(payload).encode()
    # No secret set
    resp = client.post("/confirm", data=body, headers={"content-type": "application/json", "x-signature": "deadbeef"})
    assert resp.status_code == 500
    assert resp.json().get("detail") == "CRYPTO_PAY_SECRET not set"


def test_confirm_ok(monkeypatch):
    client = TestClient(app)
    # Provide secret and mock code generation
    monkeypatch.setenv("CRYPTO_PAY_SECRET", "supersecret")
    # Monkeypatch create_redeem_code to avoid DB
    import crypto_pay_api.main as m

    monkeypatch.setattr(m, "create_redeem_code", lambda months, invoice_id: "TESTCODE")

    payload = {"invoice_id": "inv2", "plan": 3, "telegram_id": 777}
    body = json.dumps(payload, separators=(",", ":")).encode()
    sig = _sign("supersecret", body)
    resp = client.post("/confirm", data=body, headers={"content-type": "application/json", "x-signature": sig})
    assert resp.status_code == 200
    assert resp.json() == {"code": "TESTCODE"}


def test_confirm_forbidden_wrong_signature(monkeypatch):
    client = TestClient(app)
    monkeypatch.setenv("CRYPTO_PAY_SECRET", "supersecret")
    payload = {"invoice_id": "inv3", "plan": 2, "telegram_id": 555}
    body = json.dumps(payload).encode()
    # Wrong signature
    resp = client.post("/confirm", data=body, headers={"content-type": "application/json", "x-signature": "bad"})
    assert resp.status_code == 403
    assert resp.json().get("detail") == "invalid signature"

