"""Data ingestion package."""

from .ingest_news import IngestNewsInput, IngestNewsOutput
from .ingest_news import run as ingest_news_run

__all__ = ["IngestNewsInput", "IngestNewsOutput", "ingest_news_run"]
