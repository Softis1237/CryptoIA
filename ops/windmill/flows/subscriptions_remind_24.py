#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> None:
    if "/app/src" not in sys.path:
        sys.path.append("/app/src")
    from pipeline.telegram_bot.subscriptions import send_renew_reminders

    c = send_renew_reminders(hours_before=24)
    print(f"reminders={c}")


if __name__ == "__main__":
    main()
