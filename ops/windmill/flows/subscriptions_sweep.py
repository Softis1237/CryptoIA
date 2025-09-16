#!/usr/bin/env python3
from __future__ import annotations

import sys


def main():
    if "/app/src" not in sys.path:
        sys.path.append("/app/src")
    from pipeline.telegram_bot.subscriptions import sweep_and_revoke_channel_access  # type: ignore

    c = sweep_and_revoke_channel_access()
    print(f"expired={c}")


if __name__ == "__main__":
    main()
