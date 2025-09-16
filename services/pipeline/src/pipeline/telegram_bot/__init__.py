"""Пакет Telegram-бота CryptoIA.

Единая точка экспорта ключевых утилит и локализации.
"""

from .messages import _t, get_message
from .publisher import (
    publish_code_block_json,
    publish_message,
    publish_message_to,
    publish_photo_from_s3,
)
from .storage import load_user_settings, save_user_setting
from .subscriptions import (
    redeem_code_and_activate,
    send_renew_reminders,
    sweep_and_revoke_channel_access,
)

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
    # Публикации и подписки
    "publish_message",
    "publish_message_to",
    "publish_photo_from_s3",
    "publish_code_block_json",
    "sweep_and_revoke_channel_access",
    "send_renew_reminders",
    "redeem_code_and_activate",
]


def main() -> None:
    from .bot import main as _main

    _main()


def build_main_menu():
    from .bot import build_main_menu as _build

    return _build()
