"""Пакет Telegram-бота CryptoIA.

Единая точка экспорта ключевых утилит и локализации.
"""

from .bot import build_main_menu, main
from .messages import _t, get_message
from .storage import load_user_settings, save_user_setting

__all__ = [
    # Основные entrypoints
    "main",
    "build_main_menu",
    # Хранилище пользовательских настроек
    "load_user_settings",
    "save_user_setting",
    # Локализация (для обратной совместимости и gettext)
    "_t",
    "get_message",
]
