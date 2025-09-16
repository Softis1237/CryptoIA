"""Телеграм-бот для сигналов и новостей."""

from __future__ import annotations

from datetime import datetime
from typing import List

from loguru import logger

try:  # pragma: no cover - provides lightweight fallback for tests without telegram
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update  # type: ignore
    from telegram.ext import (  # type: ignore
        Application,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
    )
except ModuleNotFoundError:  # pragma: no cover
    class InlineKeyboardButton:  # minimal stub for tests
        def __init__(self, text: str, callback_data: str | None = None):
            self.text = text
            self.callback_data = callback_data

    class InlineKeyboardMarkup:  # minimal stub for tests
        def __init__(self, keyboard):
            self.inline_keyboard = keyboard

    class _DummyContext:
        DEFAULT_TYPE = object

    ContextTypes = _DummyContext()  # type: ignore[assignment]

    def _missing(*_, **__):  # pragma: no cover - fail only when actual bot run
        raise RuntimeError("python-telegram-bot is required to run the Telegram bot")

    class Application:  # type: ignore[assignment]
        @staticmethod
        def builder():
            _missing()

    CallbackQueryHandler = CommandHandler = _missing  # type: ignore[assignment]
    Update = object  # type: ignore[assignment]

from ..infra.config import settings
from ..infra.db import get_conn
from .publisher import publish_message_to
from .storage import load_user_settings, save_user_setting


def build_main_menu() -> InlineKeyboardMarkup:
    keyboard: List[List[InlineKeyboardButton]] = [
        [InlineKeyboardButton("📊 Сигналы", callback_data="signals")],
        [InlineKeyboardButton("📰 Новости", callback_data="news")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")],
    ]
    return InlineKeyboardMarkup(keyboard)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Добро пожаловать в CryptoIA!",
        reply_markup=build_main_menu(),
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Доступные команды:\n"
        "/start — меню\n"
        "/help — помощь\n"
        "/settings — настройки"
    )
    if update.message:
        await update.message.reply_text(text)
    else:
        await update.callback_query.edit_message_text(text)


def _latest_signal_text() -> str:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT created_at, side, entry_zone, sl, tp "
                    "FROM trades_suggestions ORDER BY created_at DESC LIMIT 1"
                )
                row = cur.fetchone()
                if not row:
                    return "Сигналов пока нет"
                created, side, entry_zone, sl, tp = row
                ez = (
                    entry_zone
                    if isinstance(entry_zone, str)
                    else str(entry_zone)  # noqa: E501
                )
                ts = (
                    created.strftime("%Y-%m-%d %H:%M")
                    if isinstance(created, datetime)
                    else str(created)
                )
                return (
                    f"Последний сигнал ({ts}):\n{side}\n"
                    f"Entry: {ez}\nSL: {sl}\nTP: {tp}"
                )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"failed to load signal: {e}")
        return "Ошибка получения сигнала"


async def show_settings(update: Update) -> None:
    user_id = update.effective_user.id if update.effective_user else 0
    prefs = load_user_settings(user_id)
    text = (
        "Настройки:\n"
        f"Язык: {prefs['lang']}\n"
        f"Частота уведомлений: {prefs['freq']}"
    )
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("RU", callback_data="set_lang_ru"),
                InlineKeyboardButton("EN", callback_data="set_lang_en"),
            ],
            [
                InlineKeyboardButton("Часто", callback_data="set_freq_high"),
                InlineKeyboardButton("Редко", callback_data="set_freq_low"),
            ],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
        ]
    )
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(
            text,
            reply_markup=keyboard,
        )
    else:
        if update.message:
            await update.message.reply_text(
                text,
                reply_markup=keyboard,
            )


async def button_cb(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:  # noqa: E501
    q = update.callback_query
    if not q:
        return
    await q.answer()
    data = q.data or ""
    uid = q.from_user.id
    if data == "signals":
        publish_message_to(str(q.message.chat_id), _latest_signal_text())  # noqa: E501
    elif data == "news":
        await q.edit_message_text(
            "Новостной раздел в разработке",
            reply_markup=build_main_menu(),
        )
    elif data == "settings":
        await show_settings(update)
    elif data == "help":
        await help_cmd(update, context)
    elif data == "back":
        await q.edit_message_text(
            "Главное меню",
            reply_markup=build_main_menu(),
        )
    elif data.startswith("set_lang_"):
        lang = data.split("_", 2)[2]
        save_user_setting(uid, "lang", lang)
        await q.edit_message_text(
            "Язык обновлён",
            reply_markup=build_main_menu(),
        )
    elif data.startswith("set_freq_"):
        freq = data.split("_", 2)[2]
        save_user_setting(uid, "freq", freq)
        await q.edit_message_text(
            "Частота обновлена",
            reply_markup=build_main_menu(),
        )


async def settings_cmd(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:  # noqa: E501
    await show_settings(update)


def main() -> None:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    app = Application.builder().token(settings.telegram_bot_token).build()  # noqa: E501
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CallbackQueryHandler(button_cb))
    app.run_polling()


__all__ = ["main", "build_main_menu"]
