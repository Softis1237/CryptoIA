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
    secret = os.getenv("CRYPTO_PAY_SECRET", "")
    mac = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, signature)


@app.post("/confirm")
async def confirm_payment(
    request: Request, x_signature: str = Header("")
):
    body = await request.body()
    if not _verify_signature(body, x_signature):
        raise HTTPException(status_code=403, detail="invalid signature")
    payload = await request.json()
    invoice_id = str(payload.get("invoice_id"))
    plan = int(payload.get("plan", 1))
    code = create_redeem_code(plan, invoice_id)
    logger.info(f"Invoice {invoice_id} confirmed for {payload.get('telegram_id')}")
    return {"code": code}
