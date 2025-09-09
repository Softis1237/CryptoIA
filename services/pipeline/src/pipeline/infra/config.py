from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    postgres_user: str = os.getenv("POSTGRES_USER", "app")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "app")
    postgres_db: str = os.getenv("POSTGRES_DB", "crypto")
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))

    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))

    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    s3_bucket: str = os.getenv("S3_BUCKET", "artifacts")

    timezone: str = os.getenv("TIMEZONE", "Asia/Jerusalem")

    telegram_bot_token: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = os.getenv("TELEGRAM_CHAT_ID")


settings = Settings()
