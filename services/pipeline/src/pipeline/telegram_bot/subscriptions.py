from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import List

from loguru import logger

from ..infra.db import (
    add_subscription,
    get_conn,
    redeem_code_use,
    sweep_expired_subscriptions,
)
from .publisher import publish_message_to


def sweep_and_revoke_channel_access() -> int:
    """Expire subscriptions in DB and remove users from private channel."""
    channel_id = os.getenv("TELEGRAM_PRIVATE_CHANNEL_ID")
    if not channel_id:
        logger.warning(
            "TELEGRAM_PRIVATE_CHANNEL_ID not set; sweep will only update DB statuses"
        )
    expired_count = sweep_expired_subscriptions()
    if not channel_id or expired_count == 0:
        return expired_count

    users: List[int] = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT user_id FROM subscriptions WHERE status='expired'"
                " ORDER BY ends_at DESC LIMIT 1000"
            )
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
                bot.unban_chat_member(chat_id=channel_id, user_id=uid)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to revoke user {uid}: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Telegram revoke failed: {exc}")
    return expired_count


def redeem_code_and_activate(user_id: int, code: str) -> bool:
    """Redeem a code and activate subscription for given user."""
    try:
        months = redeem_code_use(code)
        if not months:
            return False
        add_subscription(
            user_id,
            provider="redeem_code",
            months=months,
            payload={"code": code},
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Redeem failed: {exc}")
        return False


def send_renew_reminders(hours_before: int = 24) -> int:
    """DM users whose subscription expires within next N hours."""
    owner = os.getenv("TELEGRAM_OWNER_ID")
    now = datetime.now(timezone.utc)
    soon = now + timedelta(hours=hours_before)
    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, ends_at FROM subscriptions"
                " WHERE status='active' AND ends_at BETWEEN %s AND %s",
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
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Reminder to {uid} failed: {exc}")
    if owner and count:
        publish_message_to(owner, f"Отправлено напоминаний о продлении: {count}")
    return count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("sweep")
    remind = sub.add_parser("remind")
    remind.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    if args.cmd == "sweep":
        cnt = sweep_and_revoke_channel_access()
        print(f"expired={cnt}")
    elif args.cmd == "remind":
        cnt = send_renew_reminders(hours_before=args.hours)
        print(f"reminders={cnt}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
