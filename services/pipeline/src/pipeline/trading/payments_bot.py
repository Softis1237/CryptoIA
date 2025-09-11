from __future__ import annotations

import os

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PreCheckoutQueryHandler,
    filters,
)

from ..infra.db import (
    add_subscription,
    get_subscription_status,
    insert_payment,
    payment_exists,
)
from .subscriptions import sweep_and_revoke_channel_access

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

PRIVATE_CHANNEL_ID = os.getenv(
    "TELEGRAM_PRIVATE_CHANNEL_ID"
)  # e.g. -100123456789 or @channel


CRYPTO_PAYMENT_URL = os.getenv("CRYPTO_PAYMENT_URL", "")

# Pricing/config (Stars)
PRICE_STARS_MONTH = int(os.getenv("PRICE_STARS_MONTH", "2500"))
PRICE_STARS_YEAR = int(os.getenv("PRICE_STARS_YEAR", "25000"))

# Simple i18n (RU/EN) kept in memory (user_data)
I18N = {
    "ru": {
        "start": "Привет! Выберите действие:",
        "about": (
            "О проекте: ежедневные релизы 00:00/12:00, прогнозы 4h/12h, "
            "новости и карточка сделки в приватном канале."
        ),
        "invoice_title": "Подписка BTC Forecast",
        "invoice_desc_month": "Месячная подписка на закрытый канал с прогнозами 2 раза в день. Промо-цена $25 (обычно $50).",
        "invoice_desc_year": "Годовая подписка на закрытый канал с прогнозами 2 раза в день за $250.",
        "invoice_item_month": "Подписка на месяц — $25 промо (обычно $50)",
        "invoice_item_year": "Подписка на год — $250",
        "payment_ok": "Оплата получена. Спасибо! Выдаю доступ в закрытый канал.",
        "no_active": "Нет активной подписки. Используйте /buy.",
        "status_active": "Статус: активна до {ends}",
        "status_expired": "Статус: истекла ({ends}). Используйте /buy для продления.",
        "status_none": "Подписка не найдена. Используйте /buy для оформления.",
        "lang_choose": "Выберите язык:",
        "lang_set": "Язык переключён на {lang}.",
        "lang_ru": "Русский",
        "lang_en": "English",
        "invite_fail": "Не удалось выдать инвайт автоматически, свяжитесь с администратором.",
        "not_enough_rights": "Недостаточно прав.",
        "sweep_done": "Готово. Истекших подписок: {count}",
        "crypto_link": "Или оплатите криптовалютой:",
        "crypto_pay": "Оплатить криптой",
        "start_menu_lang": "Выбор языка",
        "start_menu_pay": "Оплата",
        "start_menu_about": "Описание проекта",
        "choose_plan": "Выберите план:",
        "plan_month": "Месяц $25 промо (вместо $50)",
        "plan_year": "Год $250",
        "choose_method": "Выберите способ оплаты:",
        "method_stars": "Stars",
        "method_crypto": "Крипто-сайт",
        "payment_link": "Оплатите по ссылке: {link}",
        "redeem_usage": "Использование: /redeem <код>",
        "redeem_ok": "Код принят, подписка активирована.",
        "redeem_fail": "Неверный код.",
    },
    "en": {
        "start": "Hi! Choose an option:",
        "about": (
            "About: daily releases at 00:00/12:00 with 4h/12h forecasts, "
            "news and a trade card in a private channel."
        ),
        "invoice_title": "BTC Forecast Subscription",
        "invoice_desc_month": "Monthly access to a private channel with 2 posts/day. Promo price $25 (normally $50).",
        "invoice_desc_year": "Yearly access to a private channel with 2 posts/day for $250.",
        "invoice_item_month": "Monthly subscription — $25 promo (was $50)",
        "invoice_item_year": "Yearly subscription — $250",
        "payment_ok": "Payment received. Thank you! Granting channel access.",
        "no_active": "No active subscription. Use /buy.",
        "status_active": "Status: active until {ends}",
        "status_expired": "Status: expired ({ends}). Use /buy to renew.",
        "status_none": "Subscription not found. Use /buy.",
        "lang_choose": "Choose language:",
        "lang_set": "Language switched to {lang}.",
        "lang_ru": "Русский",
        "lang_en": "English",
        "invite_fail": "Failed to create invite link automatically, please contact admin.",
        "not_enough_rights": "Not enough rights.",
        "sweep_done": "Done. Expired subscriptions: {count}",
        "crypto_link": "Or pay with crypto:",
        "crypto_pay": "Pay with crypto",
        "start_menu_lang": "Language",
        "start_menu_pay": "Payment",
        "start_menu_about": "About project",
        "choose_plan": "Choose a plan:",
        "plan_month": "Month $25 promo (was $50)",
        "plan_year": "Year $250",
        "choose_method": "Choose payment method:",
        "method_stars": "Stars",
        "method_crypto": "Crypto site",
        "payment_link": "Pay via link: {link}",
        "redeem_usage": "Usage: /redeem <code>",
        "redeem_ok": "Code redeemed, subscription activated.",
        "redeem_fail": "Invalid code.",
    },
}


def _user_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:  # type: ignore[override]
    try:
        return context.user_data.get("lang", "ru")  # default RU
    except Exception:
        return "ru"


def _set_user_lang(update: Update, context: ContextTypes.DEFAULT_TYPE, lang: str) -> None:  # type: ignore[override]
    try:
        context.user_data["lang"] = "en" if lang == "en" else "ru"
    except Exception:
        pass


def _t(lang_code: str, key: str, **kwargs) -> str:
    return (I18N.get(lang_code, I18N["ru"]).get(key, key)).format(**kwargs)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    kb = [
        [InlineKeyboardButton(_t(lang, "start_menu_lang"), callback_data="menu:lang")],
        [InlineKeyboardButton(_t(lang, "start_menu_pay"), callback_data="menu:pay")],
        [
            InlineKeyboardButton(
                _t(lang, "start_menu_about"), callback_data="menu:about"
            )
        ],
    ]
    await update.message.reply_text(
        _t(lang, "start"), reply_markup=InlineKeyboardMarkup(kb)
    )


async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)


    prices = [LabeledPrice(label=_t(lang, "invoice_item"), amount=MONTH_STARS * 100)]
    await update.message.reply_invoice(
        title=_t(lang, "invoice_title"),
        description=_t(lang, "invoice_desc"),
        payload=PAYLOAD,
        currency="XTR",
        prices=prices,
        need_name=False,
        need_email=False,
    )


    kb = [
        [
            InlineKeyboardButton(_t(lang, "plan_month"), callback_data="plan:1"),
            InlineKeyboardButton(_t(lang, "plan_year"), callback_data="plan:12"),
        ]
    ]
    if update.message:
        await update.message.reply_text(
            _t(lang, "choose_plan"), reply_markup=InlineKeyboardMarkup(kb)
        )
    else:
        q = update.callback_query
        if q:
            await q.edit_message_text(
                _t(lang, "choose_plan"), reply_markup=InlineKeyboardMarkup(kb)
            )


async def plan_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.data:
        return
    await q.answer()
    _, months = q.data.split(":", 1)
    lang = _user_lang(update, context)
    kb = [
        [
            InlineKeyboardButton(
                _t(lang, "method_stars"), callback_data=f"pay:{months}:stars"
            )
        ],
        [
            InlineKeyboardButton(
                _t(lang, "method_crypto"), callback_data=f"pay:{months}:crypto"
            )
        ],
    ]
    await q.edit_message_text(
        _t(lang, "choose_method"), reply_markup=InlineKeyboardMarkup(kb)
    )


async def pay_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.data:
        return
    await q.answer()
    _, months, method = q.data.split(":")
    months_i = int(months)
    lang = _user_lang(update, context)
    if method == "stars":
        price = PRICE_STARS_MONTH if months_i == 1 else PRICE_STARS_YEAR
        label_key = "invoice_item_month" if months_i == 1 else "invoice_item_year"
        desc_key = "invoice_desc_month" if months_i == 1 else "invoice_desc_year"
        await q.message.reply_invoice(
            title=_t(lang, "invoice_title"),
            description=_t(lang, desc_key),
            payload=f"sub_{months_i}m",
            currency="XTR",
            prices=[LabeledPrice(label=_t(lang, label_key), amount=price)],
            need_name=False,
            need_email=False,
        )
    else:
        link = CRYPTO_PAYMENT_URL or "https://example.com/pay"
        await q.message.reply_text(_t(lang, "payment_link", link=link))


async def precheckout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.pre_checkout_query
    await query.answer(ok=True)


async def successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    sp = update.message.successful_payment
    charge_id = sp.telegram_payment_charge_id
    if payment_exists(charge_id):
        logger.info(f"Duplicate payment {charge_id} from user {user.id}")
        return
    logger.info(f"Payment successful from user {user.id}")
    lang = _user_lang(update, context)
    await update.message.reply_text(_t(lang, "payment_ok"))

    # Invite to private channel if configured and bot is admin
    if PRIVATE_CHANNEL_ID:
        try:
            link = await context.bot.create_chat_invite_link(
                chat_id=PRIVATE_CHANNEL_ID, name=f"sub-{user.id}"
            )
            await update.message.reply_text(f"Вступайте: {link.invite_link}")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Failed to create invite link: {e}")
            await update.message.reply_text(_t(lang, "invite_fail"))

    # Save subscription in DB and log payment
    try:
        months = 12 if sp.invoice_payload.endswith("12m") else 1
        payload = update.message.to_dict() if update and update.message else {}
        add_subscription(
            user.id, provider="telegram_stars", months=months, payload=payload
        )
        insert_payment(charge_id, user.id, sp.total_amount)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Failed to add subscription: {e}")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user = update.message.from_user
        st, ends_at = get_subscription_status(user.id)
        lang = _user_lang(update, context)
        if st == "active":
            await update.message.reply_text(_t(lang, "status_active", ends=ends_at))
        elif st == "expired":
            await update.message.reply_text(_t(lang, "status_expired", ends=ends_at))
        else:
            await update.message.reply_text(_t(lang, "status_none"))
    except Exception as e:  # noqa: BLE001
        logger.exception(f"status error: {e}")
        await update.message.reply_text(
            "Error. Try later."
            if _user_lang(update, context) == "en"
            else "Ошибка проверки статуса. Попробуйте позже."
        )


async def link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st, _ = get_subscription_status(update.message.from_user.id)
    lang = _user_lang(update, context)
    if st != "active":
        await update.message.reply_text(_t(lang, "no_active"))
        return
    if PRIVATE_CHANNEL_ID:
        try:
            link = await context.bot.create_chat_invite_link(
                chat_id=PRIVATE_CHANNEL_ID, name=f"sub-{update.message.from_user.id}"
            )
            await update.message.reply_text(f"Вступайте: {link.invite_link}")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Failed to create invite link: {e}")
            await update.message.reply_text(_t(lang, "invite_fail"))


async def renew(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await buy(update, context)


async def redeem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    if not context.args:
        await update.message.reply_text(_t(lang, "redeem_usage"))
        return
    code = context.args[0]
    valid = os.getenv("REDEEM_CODE")
    if valid and code == valid:
        try:
            add_subscription(
                update.message.from_user.id,
                provider="redeem",
                months=1,
                payload={"code": code},
            )
            await update.message.reply_text(_t(lang, "redeem_ok"))
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Redeem failed: {e}")
            await update.message.reply_text(_t(lang, "redeem_fail"))
    else:
        await update.message.reply_text(_t(lang, "redeem_fail"))


async def admin_sweep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    owner = os.getenv("TELEGRAM_OWNER_ID")
    if not owner or str(update.message.from_user.id) != owner:
        lang = _user_lang(update, context)
        await update.message.reply_text(_t(lang, "not_enough_rights"))
        return
    count = sweep_and_revoke_channel_access()
    lang = _user_lang(update, context)
    await update.message.reply_text(_t(lang, "sweep_done", count=count))


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    await update.message.reply_text(_t(lang, "about"))


async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    kb = [
        [
            InlineKeyboardButton(_t(lang, "lang_ru"), callback_data="lang:ru"),
            InlineKeyboardButton(_t(lang, "lang_en"), callback_data="lang:en"),
        ]
    ]
    await update.message.reply_text(
        _t(lang, "lang_choose"), reply_markup=InlineKeyboardMarkup(kb)
    )


async def lang_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.data:
        return
    await q.answer()
    _, new_lang = q.data.split(":", 1)
    _set_user_lang(update, context, new_lang)
    text = _t(new_lang, "lang_set", lang=("Русский" if new_lang == "ru" else "English"))
    try:
        await q.edit_message_text(text)
    except Exception:
        await q.message.reply_text(text)


async def menu_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.data:
        return
    await q.answer()
    action = q.data.split(":", 1)[1]
    if action == "lang":
        await lang_cmd(update, context)
    elif action == "pay":
        await buy(update, context)
    elif action == "about":
        await about(update, context)


def main():
    if not BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("buy", buy))
    app.add_handler(PreCheckoutQueryHandler(precheckout))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("link", link))
    app.add_handler(CommandHandler("renew", renew))
    app.add_handler(CommandHandler("redeem", redeem))
    app.add_handler(CommandHandler("admin_sweep", admin_sweep))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("lang", lang_cmd))
    app.add_handler(CallbackQueryHandler(lang_cb, pattern=r"^lang:(ru|en)$"))
    app.add_handler(CallbackQueryHandler(menu_cb, pattern=r"^menu:(lang|pay|about)$"))
    app.add_handler(CallbackQueryHandler(plan_cb, pattern=r"^plan:(1|12)$"))
    app.add_handler(
        CallbackQueryHandler(pay_cb, pattern=r"^pay:(1|12):(stars|crypto)$")
    )
    # successful payment is a Message update
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))
    logger.info("Payments bot started")
    app.run_polling(allowed_updates=["message", "pre_checkout_query", "callback_query"])


if __name__ == "__main__":
    main()
