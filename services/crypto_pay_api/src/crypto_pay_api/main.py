from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from fastapi import FastAPI, Header, HTTPException, Request
from loguru import logger
from pydantic import BaseModel

from pipeline.infra.db import create_redeem_code

app = FastAPI()


class InvoiceRequest(BaseModel):
    plan: int
    telegram_id: int


@app.post("/invoice")
async def create_invoice(req: InvoiceRequest):
    """Return payment address and invoice id for requested plan."""
    invoice_id = secrets.token_hex(8)
    address = f"crypto:{invoice_id}"
    return {"invoice_id": invoice_id, "address": address}


class ConfirmPayload(BaseModel):
    invoice_id: str
    plan: int
    telegram_id: int


def _verify_signature(body: bytes, signature: str) -> bool:
    """Verify HMAC signature of raw request body using CRYPTO_PAY_SECRET.

    Returns False on mismatch. If secret is not set, this is a misconfiguration
    and we signal it to the caller (handled in the endpoint).
    """
    secret = os.getenv("CRYPTO_PAY_SECRET", "")
    if not secret:
        # Signal misconfiguration to caller (endpoint will convert to 500)
        raise RuntimeError("CRYPTO_PAY_SECRET is not set")
    mac = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, signature)


@app.post("/confirm")
async def confirm_payment(
    payload: ConfirmPayload,
    request: Request,
    x_signature: str = Header("")
):
    # Validate signature over the raw body
    body = await request.body()
    try:
        if not _verify_signature(body, x_signature):
            raise HTTPException(status_code=403, detail="invalid signature")
    except RuntimeError as e:
        # Missing secret: explicit 500 to surface misconfiguration
        logger.error(str(e))
        raise HTTPException(status_code=500, detail="CRYPTO_PAY_SECRET not set") from e

    invoice_id = payload.invoice_id
    plan = int(payload.plan)
    code = create_redeem_code(plan, invoice_id)
    logger.info(f"Invoice {invoice_id} confirmed for {payload.telegram_id}")
    return {"code": code}
