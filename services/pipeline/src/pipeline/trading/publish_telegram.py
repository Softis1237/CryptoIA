from __future__ import annotations

import os
from loguru import logger

from ..infra.config import settings
from ..infra.s3 import download_bytes
from ..infra.db import list_active_subscriber_ids


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


def _dm_user_ids() -> list[str]:
    raw = os.getenv("TELEGRAM_DM_USER_IDS", "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()]


def _append_aff_footer(text: str) -> str:
    url = os.getenv("EXTERNAL_AFF_LINK_URL", "").strip()
    if not url:
        return text
    prefix = os.getenv("EXTERNAL_AFF_FOOTER_EN", "Recommended exchange:").strip()
    footer = f"\n\n{prefix} {url}"
    return (text + footer) if footer not in text else text


def publish_message(text: str) -> None:
    if not settings.telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN не задан — печатаю локально\n" + text)
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
        text = _append_aff_footer(text)
        dm_ids = _dm_user_ids()
        if settings.telegram_chat_id:
            for chunk in _chunk_text(text):
                bot.send_message(
                    chat_id=settings.telegram_chat_id,
                    text=chunk,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            logger.info("Отправлено в Telegram канал/чат")
        elif dm_ids:
            # Direct messages to specific users (must have started the bot)
            for uid in dm_ids:
                for chunk in _chunk_text(text):
                    try:
                        bot.send_message(
                            chat_id=uid,
                            text=chunk,
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"DM send failed to {uid}: {e}")
            logger.info(f"Отправлено в личку пользователям: {len(dm_ids)}")
        else:
            # Fallback: DM всем активным подписчикам из БД
            uids = list_active_subscriber_ids()
            if not uids:
                logger.warning("Нет активных подписчиков; печатаю локально\n" + text)
                print(text)
            for uid in uids:
                for chunk in _chunk_text(text):
                    try:
                        bot.send_message(
                            chat_id=uid,
                            text=chunk,
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"DM send failed to {uid}: {e}")
            logger.info(f"Отправлено активным подписчикам: {len(uids)}")
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
    if not settings.telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN не задан — пропускаю отправку фото")
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
        # append footer to caption
        caption = _append_aff_footer(caption or "")
        dm_ids = _dm_user_ids()
        if settings.telegram_chat_id:
            bot.send_photo(
                chat_id=settings.telegram_chat_id,
                photo=content,
                caption=caption or "",
                parse_mode="HTML",
            )
            logger.info("Фотография отправлена в канал/чат")
        elif dm_ids:
            for uid in dm_ids:
                try:
                    bot.send_photo(
                        chat_id=uid,
                        photo=content,
                        caption=caption or "",
                        parse_mode="HTML",
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"DM photo failed to {uid}: {e}")
            logger.info(f"Фото отправлено в личку: {len(dm_ids)}")
        else:
            # Fallback: DM всем активным подписчикам
            uids = list_active_subscriber_ids()
            if not uids:
                logger.warning("Нет активных подписчиков — пропускаю фото")
                return
            for uid in uids:
                try:
                    bot.send_photo(
                        chat_id=uid,
                        photo=content,
                        caption=caption or "",
                        parse_mode="HTML",
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"DM photo failed to {uid}: {e}")
            logger.info(f"Фото отправлено активным подписчикам: {len(uids)}")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Ошибка отправки фото в Telegram: {e}")


def publish_code_block_json(title: str, data: dict) -> None:
    import json

    text = f"<b>{title}</b>\n<code>\n{json.dumps(data, ensure_ascii=False, indent=2)}\n</code>"
    publish_message(text)
