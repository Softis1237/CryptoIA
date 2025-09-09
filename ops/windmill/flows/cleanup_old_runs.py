from __future__ import annotations

"""Windmill flow to remove old artifacts from S3."""

from datetime import datetime, timedelta, timezone
import os

from pipeline.infra.s3 import get_s3_client
from pipeline.infra.config import settings


def main(days: int | None = None, prefix: str = "runs/") -> None:
    keep_days = days or int(os.getenv("CLEANUP_KEEP_DAYS", "14"))
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    client = get_s3_client()
    token: str | None = None
    while True:
        resp = client.list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix, ContinuationToken=token) if token else client.list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix)
        for obj in resp.get("Contents", []):
            if obj.get("LastModified") and obj["LastModified"].replace(tzinfo=timezone.utc) < cutoff:
                client.delete_object(Bucket=settings.s3_bucket, Key=obj["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break


if __name__ == "__main__":
    main()
