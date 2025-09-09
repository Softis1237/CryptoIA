from __future__ import annotations

import io
from typing import Optional, Tuple

import boto3
from botocore.client import Config
from loguru import logger

from .config import settings


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(signature_version="s3v4"),
    )


def upload_bytes(path: str, content: bytes, content_type: Optional[str] = None) -> str:
    client = get_s3_client()
    extra_args = {"ContentType": content_type} if content_type else None
    client.upload_fileobj(io.BytesIO(content), settings.s3_bucket, path, ExtraArgs=extra_args or {})
    s3_uri = f"s3://{settings.s3_bucket}/{path}"
    logger.info(f"Uploaded to {s3_uri}")
    return s3_uri


def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        # treat as key in default bucket
        return settings.s3_bucket, s3_uri
    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    return bucket, key


def download_bytes(s3_uri_or_key: str) -> bytes:
    client = get_s3_client()
    bucket, key = _parse_s3_uri(s3_uri_or_key)
    buf = io.BytesIO()
    client.download_fileobj(bucket, key, buf)
    return buf.getvalue()
