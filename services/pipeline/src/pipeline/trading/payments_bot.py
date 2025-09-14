from __future__ import annotations

import os
from loguru import logger
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    PreCheckoutQueryHandler,
    filters,
)

from ..infra.db import ensure_payments_table, ensure_subscriptions_tables
from ..infra.health import start_background as start_health_server
from ..community.insights import evaluate_and_store_insight  # required by menus.handle_text
from .subscriptions import redeem_code_and_activate, sweep_and_revoke_channel_access  # noqa: F401
from ..telegram_bot.menus import (
    start,
    buy,
    profile,
    precheckout,
    status,
    link,
    renew,
    redeem,
    help_cmd,
    menu_router,
    intro_start_cb,
    plan_cb,
    pay_cb,
    about,
    lang_cmd,
    lang_cb,
    menu_cb,
    settings_cmd,
    settings_cb,
    post_init,
    successful_payment,
    ping_cmd,
    id_cmd,
    handle_text,
    unknown_cmd,
    error_handler,
)
from ..telegram_bot.affiliates import (
    aff_set,
    aff_stats,
    aff_approve,
    aff_list,
    affiliate_cb,
    affrequests,
    affmark,
)
from ..telegram_bot.admin import admin_cb, admin_sweep, genpromo, handle_admin_files

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


def main():
    if not BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    try:
        ensure_subscriptions_tables()
        ensure_payments_table()
    except Exception as e:
        logger.warning(f"DB ensure tables skipped: {e}")
    try:
        start_health_server()
    except Exception as e:
        logger.warning(f"Health server start failed: {e}")
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("buy", buy))
    app.add_handler(CommandHandler("profile", profile))
    app.add_handler(PreCheckoutQueryHandler(precheckout))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("link", link))
    app.add_handler(CommandHandler("renew", renew))
    app.add_handler(CommandHandler("redeem", redeem))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("affset", aff_set))
    app.add_handler(CommandHandler("affstats", aff_stats))
    app.add_handler(CommandHandler("affapprove", aff_approve))
    app.add_handler(CommandHandler("afflist", aff_list))
    app.add_handler(CommandHandler("genpromo", genpromo))
    app.add_handler(CommandHandler("affrequests", affrequests))
    app.add_handler(CommandHandler("affmark", affmark))
    app.add_handler(CommandHandler("admin_sweep", admin_sweep))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("lang", lang_cmd))
    app.add_handler(CommandHandler("settings", lang_cmd))
    app.add_handler(CommandHandler("menu", start))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(CallbackQueryHandler(lang_cb, pattern=r"^lang:(ru|en)$"))
    app.add_handler(CallbackQueryHandler(menu_cb, pattern=r"^menu:(lang|pay|about|affiliate)$"))
    app.add_handler(CallbackQueryHandler(menu_cb, pattern=r"^menu:(insight|link|how|quality)$"))
    app.add_handler(CallbackQueryHandler(intro_start_cb, pattern=r"^intro:start$"))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), menu_router))
    app.add_handler(CallbackQueryHandler(plan_cb, pattern=r"^plan:(1|12)$"))
    app.add_handler(CallbackQueryHandler(pay_cb, pattern=r"^pay:(1|12):(stars|crypto)$"))
    app.add_handler(CallbackQueryHandler(affiliate_cb, pattern=r"^aff:(become|stats)$"))
    app.add_handler(CallbackQueryHandler(affiliate_cb, pattern=r"^aff:(list|request)$"))
    app.add_handler(CallbackQueryHandler(affiliate_cb, pattern=r"^aff:(dash|payout)$"))
    app.add_handler(CallbackQueryHandler(admin_cb, pattern=r"^admin:.*$"))
    app.add_handler(CallbackQueryHandler(settings_cb, pattern=r"^settings:(lang|admin)$"))
    app.add_handler(CallbackQueryHandler(affiliate_cb, pattern=r"^aff:admin$"))
    app.add_handler(CallbackQueryHandler(admin_cb, pattern=r"^admin:(promo|bonuses|news).*$"))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
    app.add_handler(MessageHandler((filters.Document.ALL | filters.PHOTO) & (~filters.COMMAND), handle_admin_files))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))
    app.add_error_handler(error_handler)
    logger.info("Payments bot started")
    app.run_polling(allowed_updates=["message", "pre_checkout_query", "callback_query"])


if __name__ == "__main__":
    main()

