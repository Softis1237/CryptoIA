from __future__ import annotations

from loguru import logger

from ..infra.config import settings
from ..infra.s3 import download_bytes


def _chunk_text(text: str, limit: int = 4096):
    lines = text.splitlines(True)
    chunks = []
    buf = ""
    for ln in lines:
        if len(buf) + len(ln) > limit:
            chunks.append(buf)
            buf = ""
        buf += ln
    if buf:
        chunks.append(buf)
    return chunks


def publish_message(text: str) -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning(
            "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID не заданы — печатаю локально:\n" + text
        )
        print(text)
        return
    try:
        from .subscriptions import sweep_and_revoke_channel_access

        sweep_and_revoke_channel_access()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to sweep expired subscriptions: {e}")
    try:
        from telegram import Bot

        bot = Bot(token=settings.telegram_bot_token)
        for chunk in _chunk_text(text):
            bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=chunk,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        logger.info("Отправлено в Telegram")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Ошибка публикации в Telegram: {e}")


def publish_message_to(chat_id: str, text: str) -> None:
    if not settings.telegram_bot_token or not chat_id:
        logger.warning(
            "TELEGRAM_BOT_TOKEN/CHAT_ID не заданы — печатаю локально:\n" + text
        )
        print(text)
        return
    try:
        from telegram import Bot

        bot = Bot(token=settings.telegram_bot_token)
        for chunk in _chunk_text(text):
            bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        logger.info(f"Отправлено в Telegram chat {chat_id}")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Ошибка публикации в Telegram (target): {e}")


def publish_photo_from_s3(s3_uri: str, caption: str | None = None) -> None:
    if not s3_uri:
        return
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning(
            "TELEGRAM_* не заданы — пропускаю отправку фото, путь: " + s3_uri
        )
        return
    try:
        from .subscriptions import sweep_and_revoke_channel_access

        sweep_and_revoke_channel_access()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to sweep expired subscriptions: {e}")
    try:
        from telegram import Bot

        content = download_bytes(s3_uri)
        bot = Bot(token=settings.telegram_bot_token)
        bot.send_photo(
            chat_id=settings.telegram_chat_id,
            photo=content,
            caption=caption or "",
            parse_mode="HTML",
        )
        logger.info("Фотография отправлена в Telegram")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Ошибка отправки фото в Telegram: {e}")


def publish_code_block_json(title: str, data: dict) -> None:
    import json

    text = f"<b>{title}</b>\n<code>\n{json.dumps(data, ensure_ascii=False, indent=2)}\n</code>"
    publish_message(text)
