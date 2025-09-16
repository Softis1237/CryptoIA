# flake8: noqa
from __future__ import annotations

import json
import os
from typing import Iterable

from loguru import logger

from ..infra.config import settings
from ..infra.db import list_active_subscriber_ids
from ..infra.s3 import download_bytes
from .messages import get_message


def _chunk_text(text: str, limit: int = 4096) -> list[str]:
    lines = text.splitlines(True)
    chunks: list[str] = []
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


def _append_aff_footer(text: str, lang: str = "en") -> str:
    url = os.getenv("EXTERNAL_AFF_LINK_URL", "").strip()
    if not url:
        return text
    if lang == "ru":
        prefix = os.getenv(
            "EXTERNAL_AFF_FOOTER_RU",
            "Рекомендуемая биржа:",
        ).strip()
    else:
        prefix = os.getenv(
            "EXTERNAL_AFF_FOOTER_EN",
            "Recommended exchange:",
        ).strip()
    footer = f"\n\n{prefix} {url}"
    return text if footer in text else text + footer


def _ensure_swept() -> None:
    try:
        from .subscriptions import sweep_and_revoke_channel_access

        sweep_and_revoke_channel_access()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to sweep expired subscriptions: {exc}")


def _send_chunks(bot, chat_id: str, chunks: Iterable[str]) -> None:
    for chunk in chunks:
        bot.send_message(
            chat_id=chat_id,
            text=chunk,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )


def publish_message(
    text: str,
    lang: str = "en",
    append_aff: bool = True,
    **kwargs,
) -> None:
    message = get_message(text, **kwargs)
    if append_aff:
        message = _append_aff_footer(message, lang)
    if not settings.telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN не задан — печатаю локально\n" + message)
        print(message)
        return

    _ensure_swept()

    try:
        from telegram import Bot

        bot = Bot(token=settings.telegram_bot_token)
        dm_ids = _dm_user_ids()
        if settings.telegram_chat_id:
            _send_chunks(bot, settings.telegram_chat_id, _chunk_text(message))
            logger.info("Отправлено в Telegram канал/чат")
            return
        if dm_ids:
            for uid in dm_ids:
                try:
                    _send_chunks(bot, uid, _chunk_text(message))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"DM send failed to {uid}: {exc}")
            logger.info(f"Отправлено в личку пользователям: {len(dm_ids)}")
            return
        # Fallback — каждому активному подписчику
        uids = list_active_subscriber_ids()
        if not uids:
            logger.warning("Нет активных подписчиков — печатаю локально\n" + message)
            print(message)
            return
        for uid in uids:
            try:
                _send_chunks(bot, uid, _chunk_text(message))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"DM send failed to {uid}: {exc}")
        logger.info(f"Отправлено активным подписчикам: {len(uids)}")
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Ошибка публикации в Telegram: {exc}")


def publish_message_to(chat_id: str, text: str, lang: str = "en", **kwargs) -> None:
    message = get_message(text, **kwargs)
    message = _append_aff_footer(message, lang)
    if not settings.telegram_bot_token or not chat_id:
        logger.warning(
            "TELEGRAM_BOT_TOKEN/CHAT_ID не заданы — печатаю локально:\n" + message
        )
        print(message)
        return
    try:
        from telegram import Bot

        bot = Bot(token=settings.telegram_bot_token)
        _send_chunks(bot, chat_id, _chunk_text(message))
        logger.info(f"Отправлено в Telegram chat {chat_id}")
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Ошибка публикации в Telegram (target): {exc}")


def publish_photo_from_s3(
    s3_uri: str,
    caption: str | None = None,
    lang: str = "en",
    **kwargs,
) -> None:
    if not s3_uri:
        return
    if not settings.telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN не задан — пропускаю отправку фото")
        return

    _ensure_swept()

    try:
        from telegram import Bot

        content = download_bytes(s3_uri)
        bot = Bot(token=settings.telegram_bot_token)
        resolved_caption = get_message(caption, **kwargs) if caption else ""
        resolved_caption = _append_aff_footer(resolved_caption, lang)
        dm_ids = _dm_user_ids()
        if settings.telegram_chat_id:
            bot.send_photo(
                chat_id=settings.telegram_chat_id,
                photo=content,
                caption=resolved_caption or "",
                parse_mode="HTML",
            )
            logger.info("Фотография отправлена в канал/чат")
            return
        if dm_ids:
            for uid in dm_ids:
                try:
                    bot.send_photo(
                        chat_id=uid,
                        photo=content,
                        caption=resolved_caption or "",
                        parse_mode="HTML",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"DM photo failed to {uid}: {exc}")
            logger.info(f"Фото отправлено в личку: {len(dm_ids)}")
            return
        uids = list_active_subscriber_ids()
        if not uids:
            logger.warning("Нет активных подписчиков — пропускаю фото")
            return
        for uid in uids:
            try:
                bot.send_photo(
                    chat_id=uid,
                    photo=content,
                    caption=resolved_caption or "",
                    parse_mode="HTML",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"DM photo failed to {uid}: {exc}")
        logger.info(f"Фото отправлено активным подписчикам: {len(uids)}")
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Ошибка отправки фото в Telegram: {exc}")


def publish_code_block_json(title: str, data: dict, lang: str = "en") -> None:
    text = (
        f"<b>{title}</b>\n<code>\n"
        f"{json.dumps(data, ensure_ascii=False, indent=2)}\n</code>"
    )
    publish_message(text, lang=lang, append_aff=False)


__all__ = [
    "publish_message",
    "publish_message_to",
    "publish_photo_from_s3",
    "publish_code_block_json",
    "_chunk_text",
    "_dm_user_ids",
    "_append_aff_footer",
]
