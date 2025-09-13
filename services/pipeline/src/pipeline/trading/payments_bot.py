from __future__ import annotations

import os

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PreCheckoutQueryHandler,
    filters,
)

import requests
from ..infra.db import (
    add_subscription,
    get_subscription_status,
    insert_payment,
    payment_exists,
    mark_payment_refunded,
    mark_subscription_refunded,
    get_or_create_affiliate,
    get_affiliate_stats,
    set_affiliate_percent,
    upsert_user_referrer,
    apply_affiliate_commission_for_first_purchase,
    get_user_referrer_info,
    list_referrals,
    insert_affiliate_request,
    list_affiliate_requests,
    mark_affiliate_request,
    get_affiliate_by_code,
)
from ..infra.metrics import push_values
from ..infra.health import start_background as start_health_server
from ..community.insights import evaluate_and_store_insight
from .subscriptions import redeem_code_and_activate, sweep_and_revoke_channel_access


BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

PRIVATE_CHANNEL_ID = os.getenv(
    "TELEGRAM_PRIVATE_CHANNEL_ID"
)  # e.g. -100123456789 or @channel



# Pricing/config (legacy MONTH/YEAR kept only for backward compat if used elsewhere)
PAYLOAD = "subscription_1m"

PROVIDER_TOKEN = os.getenv("TELEGRAM_PROVIDER_TOKEN", "")
CRYPTO_PAY_API_URL = os.getenv("CRYPTO_PAY_API_URL", "")

CRYPTO_PAYMENT_URL = os.getenv("CRYPTO_PAYMENT_URL", "")

OWNER_ID = os.getenv("TELEGRAM_OWNER_ID")
ADMIN_IDS = {
    x.strip() for x in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",") if x.strip()
}


def _is_admin(user_id: int) -> bool:
    sid = str(user_id)
    if OWNER_ID and sid == OWNER_ID:
        return True
    return sid in ADMIN_IDS


# Pricing/config (Stars)
# Defaults: 1 month = 2500 stars, 1 year = 25000 stars
# Set via env PRICE_STARS_MONTH/PRICE_STARS_YEAR
PRICE_STARS_MONTH = int(os.getenv("PRICE_STARS_MONTH", "2500"))
PRICE_STARS_YEAR = int(os.getenv("PRICE_STARS_YEAR", "25000"))
ENABLE_CRYPTO_PAY = os.getenv("ENABLE_CRYPTO_PAY", "0") in {"1", "true", "True"}

# Simple i18n (RU/EN) kept in memory (user_data)
I18N = {
    "ru": {
        "start": "Привет! Выберите действие:",
        "about": (
            "О проекте: ежедневные релизы 00:00/12:00, прогнозы 4h/12h, "
            "новости и карточка сделки в приватном канале."
        ),
        "invoice_title": "Подписка BTC Forecast",
        "invoice_desc_month": "Доступ в закрытый канал на 1 месяц. Стоимость: {m_stars}⭐.",
        "invoice_desc_year": "Доступ в закрытый канал на 12 месяцев. Стоимость: {y_stars}⭐.",
        "invoice_item_month": "Подписка на 1 месяц — {m_stars}⭐",
        "invoice_item_year": "Подписка на 12 месяцев — {y_stars}⭐",
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
        "start_menu_aff": "Партнёрка",
        "affiliate_menu": "Партнёрская программа:",
        "affiliate_become": "Стать партнёром / Моя ссылка",
        "affiliate_stats": "Моя статистика",
        "affiliate_code": "Ваш код: {code}\nСсылка: https://t.me/{bot}?start=ref_{code}\nПроцент: {percent}%",
        "affiliate_stats_text": "Рефералов: {count}\nНачислено: {amount}",
        "ref_saved": "Реферальный код сохранён: {code}",
        "affiliate_list": "Мои рефералы",
        "affiliate_list_title": "Последние рефералы:",
        "affiliate_list_empty": "Пока нет рефералов.",
        "affiliate_request": "Запросить партнёрку",
        "affiliate_request_ack": "Заявка отправлена администратору. Мы свяжемся с вами.",
        "sweep_done": "Готово. Истекших подписок: {count}",
        "crypto_link": "Или оплатите криптовалютой:",
        "crypto_pay": "Оплатить криптой",
        "start_menu_lang": "Выбор языка",
        "start_menu_pay": "Оплата",
        "start_menu_about": "Описание проекта",
        "start_menu_insight": "Инсайт",
        "choose_plan": "Выберите план:",
        "plan_month": "1 месяц — {m_stars}⭐",
        "plan_year": "12 месяцев — {y_stars}⭐",
        "choose_method": "Выберите способ оплаты:",
        "method_stars": "Stars",
        "method_crypto": "Крипто-сайт",
        "payment_link": "Оплатите по ссылке: {link}",
        "redeem_usage": "Использование: /redeem <код>",
        "redeem_ok": "Код принят, подписка активирована.",
        "redeem_fail": "Неверный код.",
        "insight_prompt": "Опишите кратко инсайт или вставьте ссылку на источник (только для активных подписчиков).",
        "insight_no_active": "Инсайты доступны только активным подписчикам. Оформите подписку через /buy.",
        "insight_ack": "Спасибо! Оценка: {verdict} (truth={truth:.2f}, fresh={fresh:.2f}). Инсайт учтём в следующих прогнозах.",

        "help": (
            "Доступные команды:\n"
            "/buy — оплата подписки\n"
            "/status — статус подписки\n"
            "/renew — продление\n"
            "/redeem <код> — промокод\n"
            "/help — эта справка"
        ),

    },
    "en": {
        "start": "Hi! Choose an option:",
        "about": (
            "About: daily releases at 00:00/12:00 with 4h/12h forecasts, "
            "news and a trade card in a private channel."
        ),
        "invoice_title": "BTC Forecast Subscription",
        "invoice_desc_month": "Private channel access for 1 month. Price: {m_stars}⭐.",
        "invoice_desc_year": "Private channel access for 12 months. Price: {y_stars}⭐.",
        "invoice_item_month": "Monthly subscription — {m_stars}⭐",
        "invoice_item_year": "Yearly subscription — {y_stars}⭐",
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
        "start_menu_aff": "Affiliate",
        "affiliate_menu": "Affiliate program:",
        "affiliate_become": "Become partner / My link",
        "affiliate_stats": "My stats",
        "affiliate_code": "Your code: {code}\nLink: https://t.me/{bot}?start=ref_{code}\nPercent: {percent}%",
        "affiliate_stats_text": "Referrals: {count}\nAccrued: {amount}",
        "ref_saved": "Referral code saved: {code}",
        "affiliate_list": "My referrals",
        "affiliate_list_title": "Latest referrals:",
        "affiliate_list_empty": "No referrals yet.",
        "affiliate_request": "Request affiliate",
        "affiliate_request_ack": "Your request has been sent to admin.",
        "sweep_done": "Done. Expired subscriptions: {count}",
        "crypto_link": "Or pay with crypto:",
        "crypto_pay": "Pay with crypto",
        "start_menu_lang": "Language",
        "start_menu_pay": "Payment",
        "start_menu_about": "About project",
        "start_menu_insight": "Insight",
        "choose_plan": "Choose a plan:",
        "plan_month": "1 month — {m_stars}⭐",
        "plan_year": "12 months — {y_stars}⭐",
        "choose_method": "Choose payment method:",
        "method_stars": "Stars",
        "method_crypto": "Crypto site",
        "payment_link": "Pay via link: {link}",
        "redeem_usage": "Usage: /redeem <code>",
        "redeem_ok": "Code redeemed, subscription activated.",
        "redeem_fail": "Invalid code.",
        "insight_prompt": "Describe your insight briefly or paste a source link (active subscribers only).",
        "insight_no_active": "Insights are available to active subscribers only. Use /buy.",
        "insight_ack": "Thanks! Verdict: {verdict} (truth={truth:.2f}, fresh={fresh:.2f}). We will incorporate it into next forecasts.",
        "help": (
            "Commands:\n"
            "/buy - buy subscription\n"
            "/status - subscription status\n"
            "/renew - renew subscription\n"
            "/redeem <code> - redeem promo\n"
            "/help - show this help"
        ),
    },
}


def _user_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:  # type: ignore[override]
    try:
        return context.user_data.get("lang", "en")  # default EN
    except Exception:
        return "en"


def _set_user_lang(update: Update, context: ContextTypes.DEFAULT_TYPE, lang: str) -> None:  # type: ignore[override]
    try:
        context.user_data["lang"] = "en" if lang == "en" else "ru"
    except Exception:
        pass


def _t(lang_code: str, key: str, **kwargs) -> str:
    return (I18N.get(lang_code, I18N["ru"]).get(key, key)).format(**kwargs)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    # Capture referral deep link
    try:
        if context.args and context.args[0].startswith("ref_"):
            code = context.args[0][4:]
            ref = get_affiliate_by_code(code)
            ref_name = ref[2] if ref else None
            upsert_user_referrer(update.message.from_user.id, code, ref_name)
            await update.message.reply_text(_t(lang, "ref_saved", code=code))
            try:
                # push event for conversion funnel (start)
                if ref:
                    partner_id = ref[0]
                    push_values(job="affiliate", values={"aff_ref_start_event": 1.0}, labels={"partner": str(partner_id)})
            except Exception:
                pass
    except Exception:
        pass
    kb = [
        [InlineKeyboardButton(_t(lang, "start_menu_lang"), callback_data="menu:lang")],
        [InlineKeyboardButton(_t(lang, "start_menu_pay"), callback_data="menu:pay")],
        [
            InlineKeyboardButton(
                _t(lang, "start_menu_about"), callback_data="menu:about"
            )
        ],
        [InlineKeyboardButton(_t(lang, "start_menu_aff"), callback_data="menu:affiliate")],
        [InlineKeyboardButton(_t(lang, "start_menu_insight"), callback_data="menu:insight")],
    ]
    await update.message.reply_text(
        _t(lang, "start"), reply_markup=InlineKeyboardMarkup(kb)
    )


async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    # Show plan selection (no immediate invoice here)
    m_stars = PRICE_STARS_MONTH
    y_stars = PRICE_STARS_YEAR
    kb = [
        [
            InlineKeyboardButton(_t(lang, "plan_month", m_stars=m_stars, y_stars=y_stars), callback_data="plan:1"),
            InlineKeyboardButton(_t(lang, "plan_year", m_stars=m_stars, y_stars=y_stars), callback_data="plan:12"),
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
        [InlineKeyboardButton(_t(lang, "method_stars"), callback_data=f"pay:{months}:stars")],
    ]
    if ENABLE_CRYPTO_PAY and CRYPTO_PAY_API_URL:
        kb.append([InlineKeyboardButton(_t(lang, "method_crypto"), callback_data=f"pay:{months}:crypto")])
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
        m_stars = PRICE_STARS_MONTH
        y_stars = PRICE_STARS_YEAR
        await q.message.reply_invoice(
            title=_t(lang, "invoice_title"),
            description=_t(lang, desc_key, m_stars=m_stars, y_stars=y_stars),
            payload=f"sub_{months_i}m",
            currency="XTR",
            prices=[LabeledPrice(label=_t(lang, label_key, m_stars=m_stars, y_stars=y_stars), amount=price)],
            need_name=False,
            need_email=False,
        )
    else:
        if not ENABLE_CRYPTO_PAY or not CRYPTO_PAY_API_URL:
            await q.message.reply_text(
                _t(lang, "payment_link", link="https://example.com/pay")
            )
            return
        try:
            resp = requests.post(
                f"{CRYPTO_PAY_API_URL.rstrip('/')}/invoice",
                json={"plan": months_i, "telegram_id": q.from_user.id},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            link = data.get("address", "")
            await q.message.reply_text(_t(lang, "payment_link", link=link))
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Invoice request failed: {e}")
            await q.message.reply_text(
                _t(lang, "payment_link", link="https://example.com/pay")
            )


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
        add_subscription(user.id, provider="telegram_stars", months=months, payload=payload)
        insert_payment(charge_id, user.id, sp.total_amount)
        apply_affiliate_commission_for_first_purchase(user.id, charge_id, sp.total_amount)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Failed to add subscription: {e}")
    # Push metrics
    try:
        # generic payment events
        push_values(job="affiliate", values={"aff_payment_event": 1.0, "aff_payment_amount": float(sp.total_amount)}, labels={"partner": "none"})
        # referred payment + commission if first
        ref_code, partner_user_id, percent, _pname = get_user_referrer_info(user.id)
        if partner_user_id and percent is not None:
            push_values(job="affiliate", values={"aff_payment_event": 1.0, "aff_payment_amount": float(sp.total_amount)}, labels={"partner": str(partner_user_id)})
            # first purchase?
            from ..infra.db import get_user_payments_count

            if get_user_payments_count(user.id) == 1:
                commission = int(round(sp.total_amount * percent / 100.0))
                push_values(job="affiliate", values={
                    "aff_first_purchase_event": 1.0,
                    "aff_commission_amount": float(commission),
                }, labels={"partner": str(partner_user_id)})
            # Optional USD approximation
            usd_rate = float(os.getenv("AFF_UNIT_TO_USD", "0"))
            if usd_rate > 0:
                push_values(job="affiliate", values={
                    "aff_payment_amount_usd": float(sp.total_amount) * usd_rate,
                }, labels={"partner": str(partner_user_id)})
                if percent is not None and get_user_payments_count(user.id) == 1:
                    push_values(job="affiliate", values={
                        "aff_commission_amount_usd": float(sp.total_amount) * usd_rate * (percent / 100.0),
                    }, labels={"partner": str(partner_user_id)})
    except Exception:
        pass


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
    if not _is_admin(update.message.from_user.id):
        await update.message.reply_text(_t(lang, "not_enough_rights"))
        return
    if not context.args:
        await update.message.reply_text(_t(lang, "redeem_usage"))
        return
    code = context.args[0]
    ok = redeem_code_and_activate(update.message.from_user.id, code)
    if ok:
        await update.message.reply_text(_t(lang, "redeem_ok"))
    else:
        await update.message.reply_text(_t(lang, "redeem_fail"))


async def refund(update: Update, context: ContextTypes.DEFAULT_TYPE):
    owner = os.getenv("TELEGRAM_OWNER_ID")
    lang = _user_lang(update, context)
    if not owner or str(update.message.from_user.id) != owner:
        await update.message.reply_text(_t(lang, "not_enough_rights"))
        return
    if not context.args:
        await update.message.reply_text("Usage: /refund <charge_id>")
        return
    charge_id = context.args[0]
    try:
        user_id = mark_payment_refunded(charge_id)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"refund db error: {e}")
        await update.message.reply_text("DB error")
        return
    if not user_id:
        await update.message.reply_text(f"Payment not found: {charge_id}")
        return
    try:
        await context.bot.refund_star_payment(
            user_id=user_id, telegram_payment_charge_id=charge_id
        )
        mark_subscription_refunded(user_id)
        logger.info(f"Refunded charge {charge_id} for user {user_id}")
        await update.message.reply_text("Refunded")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"refund api error: {e}")
        await update.message.reply_text("Refund failed")


async def admin_sweep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    if not _is_admin(update.message.from_user.id):
        await update.message.reply_text(_t(lang, "not_enough_rights"))
        return
    count = sweep_and_revoke_channel_access()
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


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    await update.message.reply_text(_t(lang, "help"))


async def aff_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    if not _is_admin(update.message.from_user.id):
        await update.message.reply_text(_t(lang, "not_enough_rights"))
        return
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /affset <partner_user_id> <percent>")
        return
    try:
        pid = int(context.args[0])
        pct = int(context.args[1])
        set_affiliate_percent(pid, pct)
        await update.message.reply_text(f"Set partner {pid} percent to {pct}%")
    except Exception as e:
        logger.exception(f"affset error: {e}")
        await update.message.reply_text("Error")


async def aff_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    try:
        if _is_admin(update.message.from_user.id) and context.args:
            pid = int(context.args[0])
        else:
            pid = update.message.from_user.id
        count, amount = get_affiliate_stats(pid)
        await update.message.reply_text(_t(lang, "affiliate_stats_text", count=count, amount=amount))
    except Exception as e:
        logger.exception(f"affstats error: {e}")
        await update.message.reply_text("Error")


async def aff_approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    if not _is_admin(update.message.from_user.id):
        await update.message.reply_text(_t(lang, "not_enough_rights"))
        return
    if not context.args:
        await update.message.reply_text("Usage: /affapprove <partner_user_id> [percent] [request_id]")
        return
    try:
        pid = int(context.args[0])
        pct = int(context.args[1]) if len(context.args) > 1 else 50
        code, percent = get_or_create_affiliate(pid, None, pct)
        if len(context.args) > 2:
            try:
                mark_affiliate_request(context.args[2], "approved")
            except Exception:
                pass
        await update.message.reply_text(f"Approved partner {pid}: code={code}, percent={percent}%")
    except Exception as e:
        logger.exception(f"affapprove error: {e}")
        await update.message.reply_text("Error")


async def aff_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    try:
        pid = update.message.from_user.id
        if _is_admin(pid) and context.args:
            pid = int(context.args[0])
        rows = list_referrals(pid, 10)
        if not rows:
            await update.message.reply_text(_t(lang, "affiliate_list_empty"))
            return
        lines = [f"• {r[0]} — amt={r[2]} comm={r[3]} at {r[4]}" for r in rows]
        await update.message.reply_text(_t(lang, "affiliate_list_title") + "\n" + "\n".join(lines))
    except Exception as e:
        logger.exception(f"afflist error: {e}")
        await update.message.reply_text("Error")


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
    elif action == "affiliate":
        await affiliate_menu(update, context)
    elif action == "insight":
        # Only prompt; next text message will be captured
        lang = _user_lang(update, context)
        try:
            context.user_data["awaiting_insight"] = True
        except Exception:
            pass
        q = update.callback_query
        if q:
            try:
                await q.edit_message_text(_t(lang, "insight_prompt"))
            except Exception:
                await q.message.reply_text(_t(lang, "insight_prompt"))


async def affiliate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    lang = _user_lang(update, context)
    kb = [
        [InlineKeyboardButton(_t(lang, "affiliate_become"), callback_data="aff:become")],
        [InlineKeyboardButton(_t(lang, "affiliate_stats"), callback_data="aff:stats")],
        [InlineKeyboardButton(_t(lang, "affiliate_list"), callback_data="aff:list")],
        [InlineKeyboardButton(_t(lang, "affiliate_request"), callback_data="aff:request")],
    ]
    if q:
        try:
            await q.edit_message_text(_t(lang, "affiliate_menu"), reply_markup=InlineKeyboardMarkup(kb))
        except Exception:
            await q.message.reply_text(_t(lang, "affiliate_menu"), reply_markup=InlineKeyboardMarkup(kb))
    else:
        await update.message.reply_text(_t(lang, "affiliate_menu"), reply_markup=InlineKeyboardMarkup(kb))


async def affiliate_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.data:
        return
    await q.answer()
    lang = _user_lang(update, context)
    action = q.data.split(":", 1)[1]
    if action == "become":
        me = await context.bot.get_me()
        code, percent = get_or_create_affiliate(q.from_user.id, q.from_user.username or q.from_user.full_name)
        text = _t(lang, "affiliate_code", code=code, bot=me.username, percent=percent)
        try:
            await q.edit_message_text(text)
        except Exception:
            await q.message.reply_text(text)
    elif action == "stats":
        count, amount = get_affiliate_stats(q.from_user.id)
        text = _t(lang, "affiliate_stats_text", count=count, amount=amount)
        try:
            await q.edit_message_text(text)
        except Exception:
            await q.message.reply_text(text)
    elif action == "list":
        rows = list_referrals(q.from_user.id, 10)
        if not rows:
            text = _t(lang, "affiliate_list_empty")
        else:
            lines = [
                f"• {r[0]} — amt={r[2]} comm={r[3]} at {r[4]}"
                for r in rows
            ]
            text = _t(lang, "affiliate_list_title") + "\n" + "\n".join(lines)
        try:
            await q.edit_message_text(text)
        except Exception:
            await q.message.reply_text(text)
    elif action == "request":
        owner = os.getenv("TELEGRAM_OWNER_ID")
        ack = _t(lang, "affiliate_request_ack")
        try:
            req_id = insert_affiliate_request(q.from_user.id, q.from_user.username, None)
            if owner:
                await context.bot.send_message(
                    chat_id=owner,
                    text=(
                        f"Affiliate request id={req_id} from @{q.from_user.username or q.from_user.id} "
                        f"(id={q.from_user.id}). Approve: /affapprove {q.from_user.id} 50 {req_id} — or mark: /affmark {req_id} approved|rejected"
                    ),
                )
        except Exception:
            pass
        try:
            await q.edit_message_text(ack)
        except Exception:
            await q.message.reply_text(ack)


async def post_init(app: Application) -> None:
    commands_en = [
        ("start", "start menu"),
        ("buy", "buy subscription"),
        ("status", "subscription status"),
        ("renew", "renew subscription"),
        ("redeem", "redeem code"),
        ("help", "show help"),
    ]
    commands_ru = [
        ("start", "стартовое меню"),
        ("buy", "оплата подписки"),
        ("status", "статус подписки"),
        ("renew", "продление"),
        ("redeem", "активация кода"),
        ("help", "подсказка"),
    ]
    await app.bot.set_my_commands(commands_en, language_code="en")
    await app.bot.set_my_commands(commands_ru, language_code="ru")


def main():
    if not BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    # Ensure required tables exist
    try:
        from ..infra.db import ensure_subscriptions_tables, ensure_payments_table

        ensure_subscriptions_tables()
        ensure_payments_table()
    except Exception as e:
        logger.warning(f"DB ensure tables skipped: {e}")
    # Health endpoint
    try:
        start_health_server()
    except Exception as e:
        logger.warning(f"Health server start failed: {e}")
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("buy", buy))
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
    
    async def affrequests(update: Update, context: ContextTypes.DEFAULT_TYPE):
        lang = _user_lang(update, context)
        if not _is_admin(update.message.from_user.id):
            await update.message.reply_text(_t(lang, "not_enough_rights"))
            return
        status = context.args[0] if context.args else "pending"
        rows = list_affiliate_requests(status=status, limit=20)
        if not rows:
            await update.message.reply_text(f"No {status} requests")
            return
        lines = [f"• {r[0]} user={r[1]} @{r[2] or ''} note={r[3] or ''} at {r[4]}" for r in rows]
        await update.message.reply_text("Requests:\n" + "\n".join(lines))

    async def affmark(update: Update, context: ContextTypes.DEFAULT_TYPE):
        lang = _user_lang(update, context)
        if not _is_admin(update.message.from_user.id):
            await update.message.reply_text(_t(lang, "not_enough_rights"))
            return
        if len(context.args) != 2:
            await update.message.reply_text("Usage: /affmark <request_id> <approved|rejected>")
            return
        try:
            mark_affiliate_request(context.args[0], context.args[1])
            await update.message.reply_text("OK")
        except Exception as e:
            logger.exception(f"affmark error: {e}")
            await update.message.reply_text("Error")

    app.add_handler(CommandHandler("affrequests", affrequests))
    app.add_handler(CommandHandler("affmark", affmark))

    app.add_handler(CommandHandler("refund", refund))

    app.add_handler(CommandHandler("admin_sweep", admin_sweep))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("lang", lang_cmd))
    app.add_handler(CallbackQueryHandler(lang_cb, pattern=r"^lang:(ru|en)$"))
    app.add_handler(CallbackQueryHandler(menu_cb, pattern=r"^menu:(lang|pay|about|affiliate)$"))
    app.add_handler(CallbackQueryHandler(menu_cb, pattern=r"^menu:(insight)$"))
    app.add_handler(CallbackQueryHandler(plan_cb, pattern=r"^plan:(1|12)$"))
    app.add_handler(
        CallbackQueryHandler(pay_cb, pattern=r"^pay:(1|12):(stars|crypto)$")
    )
    app.add_handler(CallbackQueryHandler(affiliate_cb, pattern=r"^aff:(become|stats)$"))
    app.add_handler(CallbackQueryHandler(affiliate_cb, pattern=r"^aff:(list|request)$"))
    # successful payment is a Message update
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))
    
    async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Capture free-form text if awaiting_insight is set
        try:
            if not context.user_data.get("awaiting_insight"):
                return
            # reset flag ASAP
            context.user_data["awaiting_insight"] = False
        except Exception:
            return
        lang = _user_lang(update, context)
        user = update.message.from_user
        st, _ = get_subscription_status(user.id)
        if st != "active":
            await update.message.reply_text(_t(lang, "insight_no_active"))
            return
        text = update.message.text or ""
        try:
            res = evaluate_and_store_insight(user.id, text)
            await update.message.reply_text(
                _t(
                    lang,
                    "insight_ack",
                    verdict=str(res.get("verdict")),
                    truth=float(res.get("score_truth", 0.0)),
                    fresh=float(res.get("score_freshness", 0.0)),
                )
            )
        except Exception as e:
            logger.exception(f"insight handle error: {e}")
            await update.message.reply_text("Error" if lang == "en" else "Ошибка")

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
    logger.info("Payments bot started")
    app.run_polling(allowed_updates=["message", "pre_checkout_query", "callback_query"])


if __name__ == "__main__":
    main()
