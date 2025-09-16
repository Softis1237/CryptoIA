import types

import asyncio

from pipeline.telegram_bot import build_main_menu
from pipeline.telegram_bot.bot import start


def test_main_menu_buttons():
    keyboard = build_main_menu()
    texts = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert texts == ["📊 Сигналы", "📰 Новости", "⚙️ Настройки", "ℹ️ Помощь"]


def test_start_command_replies_with_menu():
    called = {}

    class DummyMessage:
        async def reply_text(self, text, reply_markup):
            called["text"] = text
            called["markup"] = reply_markup

    update = types.SimpleNamespace(message=DummyMessage())
    asyncio.run(start(update, context=types.SimpleNamespace()))

    assert called["text"].startswith("Добро пожаловать")
    assert hasattr(called["markup"], "inline_keyboard")
    assert len(called["markup"].inline_keyboard) == 4
