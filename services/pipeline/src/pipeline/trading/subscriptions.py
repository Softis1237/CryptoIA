from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import List

from loguru import logger

from ..infra.db import get_conn, sweep_expired_subscriptions
from .publish_telegram import publish_message_to


def sweep_and_revoke_channel_access() -> int:
    """Expire subscriptions in DB and remove users from private channel."""
    channel_id = os.getenv("TELEGRAM_PRIVATE_CHANNEL_ID")
    if not channel_id:
        logger.warning("TELEGRAM_PRIVATE_CHANNEL_ID not set; sweep will only update DB statuses")
    expired_count = sweep_expired_subscriptions()
    if not channel_id or expired_count == 0:
        return expired_count
    # Fetch expired users (simple: last updates)
    users: List[int] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT user_id FROM subscriptions WHERE status='expired' ORDER BY ends_at DESC LIMIT 1000")
            rows = cur.fetchall() or []
            users = [int(r[0]) for r in rows]
    if not users:
        return expired_count
    try:
        from telegram import Bot

        bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
        for uid in users:
            try:
                bot.ban_chat_member(chat_id=channel_id, user_id=uid)
                bot.unban_chat_member(chat_id=channel_id, user_id=uid)  # ensures user removed, but can rejoin via new invite
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to revoke user {uid}: {e}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Telegram revoke failed: {e}")
    return expired_count


def send_renew_reminders(hours_before: int = 24) -> int:
    """DM users whose subscription expires within next N hours."""
    owner = os.getenv("TELEGRAM_OWNER_ID")
    now = datetime.now(timezone.utc)
    soon = now + timedelta(hours=hours_before)
    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, ends_at FROM subscriptions WHERE status='active' AND ends_at BETWEEN %s AND %s",
                (now, soon),
            )
            rows = cur.fetchall() or []
    for uid, ends_at in rows:
        try:
            text = (
                "Ваша подписка скоро истекает. Используйте /buy для продления.\n"
                f"Истекает: {str(ends_at)}"
            )
            publish_message_to(str(uid), text)
            count += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Reminder to {uid} failed: {e}")
    if owner and count:
        publish_message_to(owner, f"Отправлено напоминаний о продлении: {count}")
    return count


def main():
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("sweep")
    r = sub.add_parser("remind")
    r.add_argument("--hours", type=int, default=24)
    args = p.parse_args()

    if args.cmd == "sweep":
        c = sweep_and_revoke_channel_access()
        print(f"expired={c}")
    elif args.cmd == "remind":
        c = send_renew_reminders(hours_before=args.hours)
        print(f"reminders={c}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
