from pipeline.telegram_bot import build_main_menu


def test_main_menu_buttons():
    keyboard = build_main_menu()
    texts = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert texts == ["📊 Сигналы", "📰 Новости", "⚙️ Настройки", "ℹ️ Помощь"]
