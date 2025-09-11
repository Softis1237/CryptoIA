from __future__ import annotations

import hashlib
import hmac
import os

from fastapi import FastAPI, Header, HTTPException, Request

app = FastAPI()


def _verify_signature(body: bytes, signature: str, secret: str) -> bool:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)


@app.post("/webhook/crypto")
async def crypto_webhook(
    request: Request, x_signature: str = Header(default="")
) -> dict[str, bool]:
    secret = os.getenv("CRYPTO_WEBHOOK_SECRET", "")
    body = await request.body()
    if not secret or not _verify_signature(body, x_signature, secret):
        raise HTTPException(status_code=401, detail="invalid signature")
    # Process webhook payload here (e.g. update subscription)
    return {"ok": True}
