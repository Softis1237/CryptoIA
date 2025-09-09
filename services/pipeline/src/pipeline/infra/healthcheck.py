from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from .db import get_conn
from .s3 import get_s3_client


def run() -> bool:
    ok = True
    # DB check
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
    except Exception as e:  # noqa: BLE001
        logger.exception(f"DB health failed: {e}")
        ok = False

    # S3 check
    try:
        s3 = get_s3_client()
        s3.list_buckets()
    except Exception as e:  # noqa: BLE001
        logger.exception(f"S3 health failed: {e}")
        ok = False

    logger.info(f"healthcheck: {'OK' if ok else 'FAIL'} at {datetime.now(timezone.utc).isoformat()}")
    return ok


def main():
    run()


if __name__ == "__main__":
    main()
