
from __future__ import annotations

from ..telegram_bot.publisher import (
    publish_message_to as _publish_message_to,
    publish_photo_from_s3 as _publish_photo_from_s3,
)


def publish_message_to(chat_id: str, text: str, lang: str = "en", **kwargs) -> None:
    """Proxy to pipeline.telegram_bot.publisher.publish_message_to."""
    _publish_message_to(chat_id, text, lang=lang, **kwargs)


def publish_photo_from_s3(s3_uri: str, caption: str | None = None, **kwargs) -> None:
    """Proxy to pipeline.telegram_bot.publisher.publish_photo_from_s3."""
    _publish_photo_from_s3(s3_uri, caption, **kwargs)
