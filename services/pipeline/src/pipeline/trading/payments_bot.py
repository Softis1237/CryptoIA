from __future__ import annotations

import os

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, Update
from telegram import ReplyKeyboardMarkup
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
    get_user_payments_count,
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
    get_latest_content,
    set_content_block,
    add_news_item,
    list_news_items,
    list_content_items,
    get_affiliate_balance,
    list_user_payments,
    fetch_redeem_code,
    mark_redeem_code_used,
    set_user_discount,
    get_user_discount,
    pop_user_discount,
    create_discount_code,
    get_affiliate_for_user,
    has_pending_affiliate_request,
)
from ..infra.metrics import push_values
from ..infra.health import start_background as start_health_server
from ..community.insights import evaluate_and_store_insight
from .subscriptions import redeem_code_and_activate, sweep_and_revoke_channel_access

def _pick_banner(day: str, night: str, fallback: str) -> str:
    """Pick day/night banner by TIMEZONE env; fall back to provided key.

    Accepts S3 keys/URIs; returns selected key or empty string.
    """
    banner = fallback
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(os.getenv("TIMEZONE", "UTC"))
        hour = datetime.now(tz).hour
        if 8 <= hour < 20:
            banner = day or fallback
        else:
            banner = night or fallback
    except Exception:
        pass
    return banner


async def menu_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fallback router for plain text: map popular labels to actions.

    Provides UX similar to persistent reply keyboard buttons.
    """
    try:
        txt = (update.message.text or "").strip().lower()
        lang = _user_lang(update, context)
        def _is(s: str) -> bool:
            return s in txt
        if _is("разблокировать") or _is("unlock") or _is("подписк"):
            return await buy(update, context)
        if _is("профил") or _is("profile"):
            return await profile(update, context)
        if _is("качест") or _is("quality"):
            return await show_quality(update, context, lang)
        if _is("партн") or _is("affiliate"):
            return await affiliate_menu(update, context)
        if _is("инсайт") or _is("insight"):
            # Set awaiting flag and prompt
            try:
                context.user_data["awaiting_insight"] = True
            except Exception:
                pass
            return await (update.message.reply_text(_t(lang, "insight_prompt")))
        if _is("новост") or _is("news"):
            return await show_news(update, context, lang)
        if _is("бонус") or _is("рекоменда") or _is("bonus"):
            return await show_bonuses(update, context, lang)
        if _is("как это") or _is("how"):
            return await show_how(update, context, lang)
        if _is("промокод") or _is("promo"):
            return await show_promo(update, context, lang)
        if _is("канал") or _is("channel"):
            return await link(update, context)
        if _is("связ") or _is("support"):
            if SUPPORT_URL:
                await update.message.reply_text(SUPPORT_URL)
            else:
                await update.message.reply_text(_t(lang, "support"))
            return
        if _is("настрой") or _is("setting"):
            return await settings_cmd(update, context)
        # default → show main
        return await start(update, context)
    except Exception:
        try:
            await start(update, context)
        except Exception:
            pass

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

PRIVATE_CHANNEL_ID = os.getenv(
    "TELEGRAM_PRIVATE_CHANNEL_ID"
)  # e.g. -100123456789 or @channel



# Pricing/config (legacy MONTH/YEAR kept only for backward compat if used elsewhere)
PAYLOAD = "subscription_1m"

PROVIDER_TOKEN = os.getenv("TELEGRAM_PROVIDER_TOKEN", "")
CRYPTO_PAY_API_URL = os.getenv("CRYPTO_PAY_API_URL", "")

CRYPTO_PAYMENT_URL = os.getenv("CRYPTO_PAYMENT_URL", "")

# Branding / UX configuration
BRAND_NAME = os.getenv("BRAND_NAME", "BTC Forecast")
WELCOME_STICKER_FILE_ID = os.getenv("WELCOME_STICKER_FILE_ID", "")
WELCOME_BANNER_S3 = os.getenv("WELCOME_BANNER_S3", "")
WELCOME_BANNER_S3_DAY = os.getenv("WELCOME_BANNER_S3_DAY", "")
WELCOME_BANNER_S3_NIGHT = os.getenv("WELCOME_BANNER_S3_NIGHT", "")
# Day/Night banners for How/Quality/Promo screens
HOW_BANNER_S3 = os.getenv("HOW_BANNER_S3", "")
HOW_BANNER_S3_DAY = os.getenv("HOW_BANNER_S3_DAY", "")
HOW_BANNER_S3_NIGHT = os.getenv("HOW_BANNER_S3_NIGHT", "")
QUALITY_BANNER_S3 = os.getenv("QUALITY_BANNER_S3", "")
QUALITY_BANNER_S3_DAY = os.getenv("QUALITY_BANNER_S3_DAY", "")
QUALITY_BANNER_S3_NIGHT = os.getenv("QUALITY_BANNER_S3_NIGHT", "")
PROMO_BANNER_S3 = os.getenv("PROMO_BANNER_S3", "")
PROMO_BANNER_S3_DAY = os.getenv("PROMO_BANNER_S3_DAY", "")
PROMO_BANNER_S3_NIGHT = os.getenv("PROMO_BANNER_S3_NIGHT", "")
PUBLIC_CHANNEL_URL = os.getenv("PUBLIC_CHANNEL_URL", "")
SUPPORT_URL = os.getenv("SUPPORT_URL", "")

# Optional external affiliate/exchange link (non-intrusive CTA in affiliate menu)
EXTERNAL_AFF_LINK_URL = os.getenv("EXTERNAL_AFF_LINK_URL", "")
EXTERNAL_AFF_LINK_TEXT = os.getenv("EXTERNAL_AFF_LINK_TEXT", "💠 Бонусы")

# Branding / UX configuration
BRAND_NAME = os.getenv("BRAND_NAME", "BTC Forecast")
WELCOME_STICKER_FILE_ID = os.getenv("WELCOME_STICKER_FILE_ID", "")
WELCOME_BANNER_S3 = os.getenv("WELCOME_BANNER_S3", "")
PUBLIC_CHANNEL_URL = os.getenv("PUBLIC_CHANNEL_URL", "")
SUPPORT_URL = os.getenv("SUPPORT_URL", "")

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
        "start": "<b>Добро пожаловать в {brand}</b>\n\nВыберите раздел:",
        "kb_profile": "👤 Профиль",
        "kb_unlock": "🔓 Разблокировать PRO",
        "kb_promo": "🎁 Ввести промокод",
        "kb_quality": "📊 Качество",
        "kb_how": "📚 Как это работает",
        "kb_affiliate": "🤝 Партнёрка",
        "kb_channel": "🌐 Канал",
        "kb_support": "💬 Связаться",
        "kb_settings": "⚙️ Настройки",
        "kb_insight": "🧠 Инсайт",
        "kb_news": "📰 Новости",
        "kb_bonuses": "🎯 Бонусы и рекомендации",
        "news_title": "Последние новости:",
        "news_empty": "Пока нет новостей.",
        "bonuses_title": "Бонусы и рекомендации:",
        "bonuses_empty": "Раздел скоро будет заполнен.",
        "admin_menu": "Панель администратора:",
        "admin_promo": "🎁 Сгенерировать коды",
        "admin_edit_bonuses": "✍️ Редактировать бонусы",
        "admin_add_news": "📝 Добавить новость",
        "admin_edit_hero_a": "🖊 Hero A",
        "admin_edit_hero_b": "🖊 Hero B",
        "not_enough_rights": "Недостаточно прав.",
        "profile": (
            "Профиль: @{}\n"
            "PRO: {}\n"
            "Статус подписки: {}\n"
            "Партнёрка: рефералов {} / начислено {}"
        ),
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
        "affiliate_become": "Стать партнёром",
        "affiliate_stats": "Статистика",
        "affiliate_code": "Ваш код: {code}\nСсылка: https://t.me/{bot}?start=ref_{code}\nПроцент: {percent}%",
        "affiliate_stats_text": "Рефералов: {count}\nНачислено: {amount}",
        "ref_saved": "Реферальный код сохранён: {code}",
        "affiliate_list": "Рефералы",
        "affiliate_list_title": "Последние рефералы:",
        "affiliate_list_empty": "Пока нет рефералов.",
        "affiliate_request": "Партнёрка",
        "affiliate_request_ack": "Заявка отправлена администратору. Мы свяжемся с вами.",
        "sweep_done": "Готово. Истекших подписок: {count}",
        "crypto_link": "Или оплатите криптовалютой:",
        "crypto_pay": "Оплатить криптой",
        "start_menu_lang": "🌍 Язык",
        "start_menu_pay": "💳 Подписка",
        "start_menu_about": "📊 О прогнозах",
        "start_menu_insight": "🧠 Инсайт",
        "start_menu_channel": "🌐 Канал",
        "start_cta": "🔓 Разблокировать PRO",
        "how": "Как это работает:\n1) Ingest данных → 2) Фичи → 3) Модели/Ансамбль → 4) Сценарии/Решение → 5) Публикация и обратная связь.",
        "quality": "Качество прогнозов (sMAPE/DA) за недавний период:",
        "affiliate_dash": "Партнёрский дашборд за 30 дней:",
        "back": "⬅️ Назад",
        "promo": "🎁 Промокод: используйте /redeem <код> для активации",
        "support": "💬 Связаться",
        "start_cta": "🔓 Разблокировать PRO",
        "how": "Как это работает:\n1) Ingest данных → 2) Фичи → 3) Модели/Ансамбль → 4) Сценарии/Решение → 5) Публикация и обратная связь.",
        "quality": "Качество прогнозов (sMAPE/DA) за недавний период:",
        "affiliate_dash": "Партнёрский дашборд за 30 дней:",
        "affiliate_dash_btn": "📈 Дашборд",
        "affiliate_payout_btn": "💸 Выплата",
        "affiliate_payout_ack": "Заявка на выплату отправлена администратору.",
        "pong": "Понг",
        "your_id": "Ваш ID: {uid}",
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
        "start": "<b>Welcome to {brand}</b>\n\nChoose a section:",
        "kb_profile": "👤 Profile",
        "kb_unlock": "🔓 Unlock PRO",
        "kb_promo": "🎁 Enter promo",
        "kb_quality": "📊 Quality",
        "kb_how": "📚 How it works",
        "kb_affiliate": "🤝 Affiliate",
        "kb_channel": "🌐 Channel",
        "kb_support": "💬 Support",
        "kb_settings": "⚙️ Settings",
        "kb_insight": "🧠 Insight",
        "kb_news": "📰 News",
        "kb_bonuses": "🎯 Bonuses & Recs",
        "news_title": "Latest news:",
        "news_empty": "No news yet.",
        "bonuses_title": "Bonuses & recommendations:",
        "bonuses_empty": "This section will be filled soon.",
        "admin_menu": "Admin panel:",
        "admin_promo": "🎁 Generate codes",
        "admin_edit_bonuses": "✍️ Edit bonuses",
        "admin_add_news": "📝 Add news",
        "admin_edit_hero_a": "🖊 Hero A",
        "admin_edit_hero_b": "🖊 Hero B",
        "not_enough_rights": "Not enough rights.",
        "profile": (
            "Profile: @{}\n"
            "PRO: {}\n"
            "Subscription status: {}\n"
            "Affiliate: referrals {} / accrued {}"
        ),
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
        "affiliate_become": "Become partner",
        "affiliate_stats": "Stats",
        "affiliate_code": "Your code: {code}\nLink: https://t.me/{bot}?start=ref_{code}\nPercent: {percent}%",
        "affiliate_stats_text": "Referrals: {count}\nAccrued: {amount}",
        "ref_saved": "Referral code saved: {code}",
        "affiliate_list": "Referrals",
        "affiliate_list_title": "Latest referrals:",
        "affiliate_list_empty": "No referrals yet.",
        "affiliate_request": "Apply",
        "affiliate_request_ack": "Your request has been sent to admin.",
        "sweep_done": "Done. Expired subscriptions: {count}",
        "crypto_link": "Or pay with crypto:",
        "crypto_pay": "Pay with crypto",
        "start_menu_lang": "🌍 Language",
        "start_menu_pay": "💳 Subscription",
        "start_menu_about": "📊 About forecasts",
        "start_menu_insight": "🧠 Insight",
        "start_menu_channel": "🌐 Channel",
        "start_cta": "🔓 Unlock PRO",
        "how": "How it works:\n1) Data ingest → 2) Features → 3) Models/Ensemble → 4) Scenarios/Decision → 5) Publish & feedback.",
        "quality": "Forecast quality (sMAPE/DA) for recent period:",
        "affiliate_dash": "Affiliate dashboard (last 30 days):",
        "back": "⬅️ Back",
        "promo": "🎁 Promo: use /redeem <CODE> to activate",
        "support": "💬 Support",
        "start_cta": "🔓 Unlock PRO",
        "how": "How it works:\n1) Data ingest → 2) Features → 3) Models/Ensemble → 4) Scenarios/Decision → 5) Publish & feedback.",
        "quality": "Forecast quality (sMAPE/DA) for recent period:",
        "affiliate_dash": "Affiliate dashboard (last 30 days):",
        "affiliate_dash_btn": "📈 Dashboard",
        "affiliate_payout_btn": "💸 Payout",
        "affiliate_payout_ack": "Payout request sent to admin.",
        "pong": "Pong",
        "your_id": "Your ID: {uid}",
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
    # Ensure admin has free subscription
    try:
        uid0 = update.message.from_user.id
        if _is_admin(uid0):
            st0, _ends0 = get_subscription_status(uid0)
            if st0 != "active":
                add_subscription(uid0, provider="admin_free", months=1200, payload={"reason": "admin"})
    except Exception:
        pass
    # Intro: show once until user clicks Start
    if not context.user_data.get("onboarded"):
        try:
            # Choose banner by local time if provided
            banner = WELCOME_BANNER_S3
            try:
                from datetime import datetime
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(os.getenv("TIMEZONE", "UTC"))
                hour = datetime.now(tz).hour
                if 8 <= hour < 20:
                    banner = WELCOME_BANNER_S3_DAY or banner
                else:
                    banner = WELCOME_BANNER_S3_NIGHT or banner
            except Exception:
                pass
            if WELCOME_STICKER_FILE_ID:
                await context.bot.send_sticker(chat_id=update.effective_chat.id, sticker=WELCOME_STICKER_FILE_ID)
            elif banner:
                from ..infra.s3 import download_bytes as _dl
                content = _dl(banner)
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=content)
        except Exception:
            pass
    intro_kb = [
        [InlineKeyboardButton(_t(lang, "start_cta"), callback_data="intro:start")],
        [InlineKeyboardButton("📚 How / Как", callback_data="menu:how"), InlineKeyboardButton("📊 Quality", callback_data="menu:quality")],
        ([InlineKeyboardButton(_t(lang, "start_menu_channel"), url=PUBLIC_CHANNEL_URL)] if PUBLIC_CHANNEL_URL else [InlineKeyboardButton(_t(lang, "start_menu_channel"), callback_data="menu:link")]),
        ([InlineKeyboardButton(_t(lang, "support"), url=SUPPORT_URL)] if SUPPORT_URL else []),
        [InlineKeyboardButton(_t(lang, "promo"), callback_data="menu:promo")],
    ]
    # Persistent reply keyboard like in the example
    ENABLE_NEWS = os.getenv("ENABLE_NEWS", "1") in {"1","true","True","yes"}
    rk_rows = [
        [_t(lang, "kb_profile"), _t(lang, "kb_quality"), _t(lang, "kb_how")],
        [_t(lang, "kb_unlock"), _t(lang, "kb_promo"), _t(lang, "kb_affiliate")],
        [_t(lang, "kb_insight"), _t(lang, "kb_settings")],
        [_t(lang, "kb_bonuses")] + ([ _t(lang, "kb_news") ] if ENABLE_NEWS else []),
        [_t(lang, "kb_channel"), _t(lang, "kb_support")],
    ]
    rk = ReplyKeyboardMarkup(
        rk_rows,
        resize_keyboard=True,
        is_persistent=True,
        one_time_keyboard=False,
    )
    await update.message.reply_text(
        _t(lang, "start", brand=BRAND_NAME),
        reply_markup=rk,
        parse_mode="HTML",
    )
    # Hero A/B message (if configured)
    try:
        uid = update.message.from_user.id
        variant = 'a' if (int(uid) % 2 == 0) else 'b'
        txt_key = f"hero_{variant}"
        txt, _ = get_latest_content(txt_key)
        if txt:
            await update.message.reply_text(txt)
            try:
                push_values(job="ab", values={"hero_show": 1.0}, labels={"variant": variant})
            except Exception:
                pass
    except Exception:
        pass
    # Inline CTA message (separate) to keep both styles
    try:
        await update.message.reply_text(
            _t(lang, "promo"), reply_markup=InlineKeyboardMarkup(intro_kb)
        )
    except Exception:
        pass
        return
    # Build richer 2x3 menu (main)
    row1 = [InlineKeyboardButton(_t(lang, "start_menu_about"), callback_data="menu:about"), InlineKeyboardButton(_t(lang, "start_menu_pay"), callback_data="menu:pay")]
    row2 = [InlineKeyboardButton(_t(lang, "start_menu_aff"), callback_data="menu:affiliate"), InlineKeyboardButton(_t(lang, "start_menu_insight"), callback_data="menu:insight")]
    if PUBLIC_CHANNEL_URL:
        row3 = [InlineKeyboardButton(_t(lang, "start_menu_channel"), url=PUBLIC_CHANNEL_URL), InlineKeyboardButton(_t(lang, "start_menu_lang"), callback_data="menu:lang")]
    else:
        row3 = [InlineKeyboardButton(_t(lang, "start_menu_channel"), callback_data="menu:link"), InlineKeyboardButton(_t(lang, "start_menu_lang"), callback_data="menu:lang")]
    bk = [InlineKeyboardButton(_t(lang, "back"), callback_data="menu:main")]
    kb = [row1, row2, row3, bk]
    # Feature flags
    ENABLE_NEWS = os.getenv("ENABLE_NEWS", "1") in {"1","true","True","yes"}
    rk_rows = [
        [_t(lang, "kb_profile"), _t(lang, "kb_quality"), _t(lang, "kb_how")],
        [_t(lang, "kb_unlock"), _t(lang, "kb_promo"), _t(lang, "kb_affiliate")],
        [_t(lang, "kb_insight"), _t(lang, "kb_settings")],
        [_t(lang, "kb_bonuses")] + ([ _t(lang, "kb_news") ] if ENABLE_NEWS else []),
        [_t(lang, "kb_channel"), _t(lang, "kb_support")],
    ]
    rk = ReplyKeyboardMarkup(
        rk_rows,
        resize_keyboard=True,
        is_persistent=True,
        one_time_keyboard=False,
    )
    await update.message.reply_text(
        _t(lang, "start", brand=BRAND_NAME), reply_markup=rk, parse_mode="HTML"
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


async def profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    try:
        user = update.effective_user
        st, ends = get_subscription_status(user.id)
        pro = ("ДА" if st == "active" else "НЕТ") if lang == "ru" else ("YES" if st == "active" else "NO")
        count, amount = get_affiliate_stats(user.id)
        balance = get_affiliate_balance(user.id)
        lines = [
            ("Профиль: @{}" if lang == "ru" else "Profile: @{}").format(user.username or user.id),
            ("PRO: {}".format(pro)),
            (("Статус подписки: {}" if lang == "ru" else "Subscription status: {}").format(ends or st)),
            (("Партнёрка: рефералов {} / начислено {} / баланс {}" if lang == "ru" else "Affiliate: refs {} / accrued {} / balance {}").format(count, amount, balance)),
        ]
        # последние платежи
        pays = list_user_payments(user.id, 5)
        if pays:
            lines.append("\n" + ("Платежи:" if lang == "ru" else "Payments:"))
            for t, amt, status in pays:
                lines.append(f"• {t} — {amt} ({status})")
        text = "\n".join(lines)
        await (update.callback_query.message.reply_text(text) if update.callback_query else update.message.reply_text(text))
    except Exception:
        try:
            await (update.callback_query.message.reply_text("Error") if update.callback_query else update.message.reply_text("Error"))
        except Exception:
            pass


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


async def admin_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.data:
        return
    try:
        await q.answer()
    except Exception:
        pass
    lang = _user_lang(update, context)
    if not _is_admin(q.from_user.id):
        return await q.answer(_t(lang, "not_enough_rights"), show_alert=True)
    data = q.data
    if data == "admin:promo":
        # Offer quick presets
        kb = [
            [InlineKeyboardButton("1m x1", callback_data="admin:promo:m1c1"), InlineKeyboardButton("1m x5", callback_data="admin:promo:m1c5"), InlineKeyboardButton("1m x10", callback_data="admin:promo:m1c10")],
            [InlineKeyboardButton("3m x1", callback_data="admin:promo:m3c1"), InlineKeyboardButton("3m x5", callback_data="admin:promo:m3c5")],
            [InlineKeyboardButton("12m x1", callback_data="admin:promo:m12c1")],
        ]
        try:
            await q.edit_message_text("Promo presets:", reply_markup=InlineKeyboardMarkup(kb))
        except Exception:
            await q.message.reply_text("Promo presets:", reply_markup=InlineKeyboardMarkup(kb))
        return
    if data.startswith("admin:promo:m"):
        import re
        m = re.search(r"m(\d+)c(\d+)", data)
        months = int(m.group(1)) if m else 1
        count = int(m.group(2)) if m else 1
        from ..infra.db import create_redeem_code
        codes = []
        for _ in range(max(1, min(count, 50))):
            codes.append(create_redeem_code(months, f"admin_{q.from_user.id}"))
        txt = "\n".join([f"/redeem {c}" for c in codes])
        try:
            await q.edit_message_text(txt)
        except Exception:
            await q.message.reply_text(txt)
        return
    if data in {"admin:hero:a", "admin:hero:b"}:
        which = 'a' if data.endswith(':a') else 'b'
        try:
            context.user_data["awaiting_hero"] = which
        except Exception:
            pass
        msg = "Отправьте текст для Hero {}".format(which.upper())
        try:
            await q.edit_message_text(msg)
        except Exception:
            await q.message.reply_text(msg)
        return
    if data == "admin:bonuses":
        try:
            context.user_data["awaiting_bonuses"] = True
        except Exception:
            pass
        try:
            await q.edit_message_text("Отправьте текст для раздела ‘Бонусы’ (заменит предыдущий)")
        except Exception:
            await q.message.reply_text("Отправьте текст для раздела ‘Бонусы’ (заменит предыдущий)")
        return
    if data == "admin:bonuses_list":
        rows = list_content_items_with_id("bonus", 20)
        if not rows:
            return await q.edit_message_text("Пока пусто")
        kb = []
        for idv, content, created in rows:
            short = (content[:40] + "…") if len(content) > 40 else content
            kb.append([InlineKeyboardButton(f"🗑 {short}", callback_data=f"admin:bonus_del:{idv}")])
        return await q.edit_message_text("Удалить пункт бонусов:", reply_markup=InlineKeyboardMarkup(kb))
    if data.startswith("admin:bonus_del:"):
        _, _, item_id = data.partition(":")
        item_id = item_id.replace("bonus_del:", "")
        try:
            from ..infra.db import delete_content_item
            delete_content_item(item_id)
            await q.edit_message_text("Удалено.")
        except Exception:
            await q.edit_message_text("Ошибка удаления.")
        return
    if data == "admin:news":
        try:
            context.user_data["awaiting_news"] = True
        except Exception:
            pass
        try:
            await q.edit_message_text("Отправьте новость текстом — она появится в разделе ‘Новости’")
        except Exception:
            await q.message.reply_text("Отправьте новость текстом — она появится в разделе ‘Новости’")
        return
    if data == "admin:news_list":
        rows = list_content_items_with_id("news", 20)
        if not rows:
            return await q.edit_message_text("Нет новостей")
        kb = []
        for idv, content, created in rows:
            short = (content[:40] + "…") if len(content) > 40 else content
            kb.append([InlineKeyboardButton(f"🗑 {short}", callback_data=f"admin:news_del:{idv}")])
        return await q.edit_message_text("Удалить новость:", reply_markup=InlineKeyboardMarkup(kb))
    if data.startswith("admin:news_del:"):
        _, _, item_id = data.partition(":")
        item_id = item_id.replace("news_del:", "")
        try:
            from ..infra.db import delete_content_item
            delete_content_item(item_id)
            await q.edit_message_text("Удалено.")
        except Exception:
            await q.edit_message_text("Ошибка удаления.")
        return


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
        # Apply discount if any
        try:
            disc = get_user_discount(q.from_user.id)
            if disc and disc > 0:
                price = max(1, int(round(price * (100 - int(disc)) / 100.0)))
        except Exception:
            pass
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


async def intro_start_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Mark onboarded and show main menu
    try:
        context.user_data["onboarded"] = True
    except Exception:
        pass
    try:
        await start(update, context)
    except Exception:
        pass


async def show_quality(update: Update, context: ContextTypes.DEFAULT_TYPE, lang: str) -> None:
    from io import BytesIO
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime
    # Pull recent outcomes and compute rolling sMAPE/DA (7/14d)
    try:
        from ..infra.db import get_conn
        rows: list[tuple[datetime, str, float | None, bool | None]] = []
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at, horizon, error_pct, direction_correct
                    FROM prediction_outcomes
                    WHERE created_at >= now() - interval '60 days'
                    ORDER BY created_at ASC
                    """
                )
                for created_at, hz, err, dc in cur.fetchall() or []:
                    rows.append((created_at, str(hz), (float(err) if err is not None else None), (bool(dc) if dc is not None else None)))
        if not rows:
            raise RuntimeError('no outcomes')
        df = pd.DataFrame(rows, columns=["ts", "hz", "err", "dc"]).dropna(subset=["ts", "hz"]).copy()
        # daily aggregation per horizon
        def _agg(dfh: pd.DataFrame) -> pd.DataFrame:
            d = dfh.set_index("ts").resample("1D").agg({"err": "mean", "dc": "mean"}).dropna(how="all")
            d["smape7"] = d["err"].rolling(7, min_periods=3).mean()
            d["smape14"] = d["err"].rolling(14, min_periods=5).mean()
            d["da7"] = d["dc"].rolling(7, min_periods=3).mean()
            d["da14"] = d["dc"].rolling(14, min_periods=5).mean()
            return d
        d4 = _agg(df[df["hz"] == "4h"]) if (df["hz"] == "4h").any() else pd.DataFrame()
        d12 = _agg(df[df["hz"] == "12h"]) if (df["hz"] == "12h").any() else pd.DataFrame()
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), dpi=200, sharex=True)
        ax1, ax2 = axes
        # sMAPE rolling
        if not d4.empty:
            ax1.plot(d4.index, d4["smape7"], label='4h sMAPE 7d', color='#1f77b4')
            ax1.plot(d4.index, d4["smape14"], label='4h sMAPE 14d', color='#1f77b4', linestyle='--', alpha=0.7)
        if not d12.empty:
            ax1.plot(d12.index, d12["smape7"], label='12h sMAPE 7d', color='#ff7f0e')
            ax1.plot(d12.index, d12["smape14"], label='12h sMAPE 14d', color='#ff7f0e', linestyle='--', alpha=0.7)
        ax1.set_title('Rolling sMAPE (lower is better)')
        ax1.set_ylabel('sMAPE, %')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        # DA rolling
        if not d4.empty:
            ax2.plot(d4.index, d4["da7"], label='4h DA 7d', color='#2ca02c')
            ax2.plot(d4.index, d4["da14"], label='4h DA 14d', color='#2ca02c', linestyle='--', alpha=0.7)
        if not d12.empty:
            ax2.plot(d12.index, d12["da7"], label='12h DA 7d', color='#9467bd')
            ax2.plot(d12.index, d12["da14"], label='12h DA 14d', color='#9467bd', linestyle='--', alpha=0.7)
        ax2.set_title('Directional Accuracy (share correct; higher is better)')
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        import matplotlib.pyplot as _plt
        _plt.close(fig)
        buf.seek(0)
        photo = buf.getvalue()
        q = update.callback_query
        caption = _t(lang, 'quality') + ("\n* sMAPE — симметричная ошибка в %, ниже лучше; DA — доля верных направлений." if lang == 'ru' else "\n* sMAPE = symmetric error %, lower is better; DA = directional accuracy.")
        try:
            await context.bot.send_photo(chat_id=(q.message.chat.id if q else update.effective_chat.id), photo=photo, caption=caption)
        except Exception:
            if q:
                await q.message.reply_text(caption)
            else:
                await update.message.reply_text(caption)
    except Exception:
        # Fallback to text only
        text = _t(lang, 'quality')
        try:
            if update.callback_query:
                await update.callback_query.message.reply_text(text)
            else:
                await update.message.reply_text(text)
        except Exception:
            pass


async def show_bonuses(update: Update, context: ContextTypes.DEFAULT_TYPE, lang: str) -> None:
    rows = list_content_items("bonus", 10)
    texts = [r[0] for r in rows] if rows else []
    if EXTERNAL_AFF_LINK_URL:
        texts.insert(0, f"{EXTERNAL_AFF_LINK_TEXT}: {EXTERNAL_AFF_LINK_URL}")
    if not texts:
        texts = [_t(lang, "bonuses_empty")]
    # Send as separate messages for CTR
    try:
        chat_id = update.callback_query.message.chat.id if update.callback_query else update.effective_chat.id
        await context.bot.send_message(chat_id=chat_id, text=_t(lang, "bonuses_title"))
        for t in texts[:5]:
            await context.bot.send_message(chat_id=chat_id, text=t, disable_web_page_preview=False)
        # Admin edit shortcut
        try:
            uid = (update.callback_query.from_user.id if update.callback_query else update.effective_user.id)
            if _is_admin(uid):
                await context.bot.send_message(chat_id=chat_id, text=("Отправьте новый пункт бонусов (каждый пункт отдельным сообщением)" if lang=="ru" else "Send a new bonus item"), reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(_t(lang, "admin_edit_bonuses"), callback_data="admin:bonuses")]]))
        except Exception:
            pass
    except Exception:
        pass


async def show_news(update: Update, context: ContextTypes.DEFAULT_TYPE, lang: str) -> None:
    rows = list_news_items(5)
    chat_id = update.callback_query.message.chat.id if update.callback_query else update.effective_chat.id
    if not rows:
        try:
            await context.bot.send_message(chat_id=chat_id, text=_t(lang, "news_empty"))
        except Exception:
            pass
    else:
        try:
            await context.bot.send_message(chat_id=chat_id, text=_t(lang, "news_title"))
        except Exception:
            pass
        for content, created in rows:
            try:
                if isinstance(content, str) and (content.startswith("DOC:") or content.startswith("PHOTO:")):
                    # Format: KIND:s3_uri|caption
                    try:
                        kind, rest = content.split(":", 1)
                        s3uri, cap = (rest.split("|", 1) + [""])[:2]
                    except Exception:
                        kind, s3uri, cap = "DOC", content[4:], ""
                    from ..infra.s3 import download_bytes as _dl
                    blob = _dl(s3uri)
                    if kind.startswith("PHOTO"):
                        await context.bot.send_photo(chat_id=chat_id, photo=blob, caption=cap)
                    else:
                        await context.bot.send_document(chat_id=chat_id, document=blob, caption=cap)
                else:
                    txt = (f"[{created}]:\n{content}" if created else content)
                    await context.bot.send_message(chat_id=chat_id, text=txt)
            except Exception:
                continue
    # Admin quick action
    try:
        uid = (update.callback_query.from_user.id if update.callback_query else update.effective_user.id)
        if _is_admin(uid):
            kb = [[InlineKeyboardButton(_t(lang, "admin_add_news"), callback_data="admin:news")]]
            await context.bot.send_message(chat_id=chat_id, text=("Добавить новость?" if lang=="ru" else "Add news?"), reply_markup=InlineKeyboardMarkup(kb))
    except Exception:
        pass


async def show_how(update: Update, context: ContextTypes.DEFAULT_TYPE, lang: str) -> None:
    """Send 'How it works' card with optional day/night banner."""
    text = _t(lang, "how")
    _banner = _pick_banner(HOW_BANNER_S3_DAY, HOW_BANNER_S3_NIGHT, HOW_BANNER_S3)
    try:
        if _banner:
            from ..infra.s3 import download_bytes as _dl
            content = _dl(_banner)
            if update.callback_query:
                await context.bot.send_photo(chat_id=update.callback_query.message.chat.id, photo=content, caption=text)
            else:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=content, caption=text)
        else:
            if update.callback_query:
                await update.callback_query.message.reply_text(text)
            else:
                await update.message.reply_text(text)
    except Exception:
        try:
            if update.callback_query:
                await update.callback_query.message.reply_text(text)
            else:
                await update.message.reply_text(text)
        except Exception:
            pass


async def show_promo(update: Update, context: ContextTypes.DEFAULT_TYPE, lang: str) -> None:
    """Send Promo instructions with optional banner."""
    text = _t(lang, "promo")
    _banner = _pick_banner(PROMO_BANNER_S3_DAY, PROMO_BANNER_S3_NIGHT, PROMO_BANNER_S3)
    try:
        if _banner:
            from ..infra.s3 import download_bytes as _dl
            content = _dl(_banner)
            if update.callback_query:
                await context.bot.send_photo(chat_id=update.callback_query.message.chat.id, photo=content, caption=text)
            else:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=content, caption=text)
        else:
            if update.callback_query:
                await update.callback_query.message.reply_text(text)
            else:
                await update.message.reply_text(text)
    except Exception:
        try:
            if update.callback_query:
                await update.callback_query.message.reply_text(text)
            else:
                await update.message.reply_text(text)
        except Exception:
            pass
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
        # consume discount if set
        try:
            pop_user_discount(user.id)
        except Exception:
            pass
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


async def link_inline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Create and show channel invite via callback if available."""
    q = update.callback_query
    if not q:
        return
    try:
        await q.answer()
    except Exception:
        pass
    lang = _user_lang(update, context)
    try:
        if PRIVATE_CHANNEL_ID:
            link = await context.bot.create_chat_invite_link(chat_id=PRIVATE_CHANNEL_ID, name=f"sub-{q.from_user.id}")
            await q.edit_message_text(f"Вступайте: {link.invite_link}")
        elif PUBLIC_CHANNEL_URL:
            await q.edit_message_text(PUBLIC_CHANNEL_URL)
        else:
            await q.edit_message_text(_t(lang, "invite_fail"))
    except Exception:
        await q.edit_message_text(_t(lang, "invite_fail"))


async def renew(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await buy(update, context)


async def redeem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    if not context.args:
        await update.message.reply_text(_t(lang, "redeem_usage"))
        return
    code = context.args[0]
    try:
        # Discount code?
        m, d, used = fetch_redeem_code(code)
        if used:
            await update.message.reply_text(_t(lang, "redeem_fail"))
            return
        if d and d > 0:
            set_user_discount(update.message.from_user.id, int(d))
            mark_redeem_code_used(code)
            await update.message.reply_text(("Скидка {}% будет применена при следующей оплате" if lang=="ru" else "Discount {}% will be applied on next purchase").format(int(d)))
            return
        # Months code
        months = redeem_code_and_activate(code, update.message.from_user.id)
        if months:
            await update.message.reply_text(_t(lang, "redeem_ok"))
        else:
            await update.message.reply_text(_t(lang, "redeem_fail"))
    except Exception as e:
        logger.exception(f"redeem error: {e}")
        await update.message.reply_text("Error")


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
    text = _t(lang, "about")
    q = getattr(update, "callback_query", None)
    if q:
        try:
            await q.edit_message_text(text)
            return
        except Exception:
            try:
                await q.message.reply_text(text)
                return
            except Exception:
                pass
    # fallback to message
    try:
        await update.message.reply_text(text)
    except Exception:
        pass


async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    kb = [
        [
            InlineKeyboardButton(_t(lang, "lang_ru"), callback_data="lang:ru"),
            InlineKeyboardButton(_t(lang, "lang_en"), callback_data="lang:en"),
        ]
    ]
    q = getattr(update, "callback_query", None)
    if q:
        try:
            await q.edit_message_text(_t(lang, "lang_choose"), reply_markup=InlineKeyboardMarkup(kb))
            return
        except Exception:
            try:
                await q.message.reply_text(_t(lang, "lang_choose"), reply_markup=InlineKeyboardMarkup(kb))
                return
            except Exception:
                pass
    try:
        await update.message.reply_text(_t(lang, "lang_choose"), reply_markup=InlineKeyboardMarkup(kb))
    except Exception:
        pass


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


async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    await update.message.reply_text(_t(lang, "pong"))


async def id_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    uid = update.message.from_user.id if update and update.message else 0
    await update.message.reply_text(_t(lang, "your_id", uid=uid))


async def unknown_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Fallback for unknown commands — show help
    lang = _user_lang(update, context)
    try:
        await update.message.reply_text(_t(lang, "help"))
    except Exception:
        pass


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[override]
    try:
        logger.exception("Bot error", exc_info=context.error)
    except Exception:
        pass


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
    try:
        await q.answer()
    except Exception:
        pass
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
    elif action == "how":
        # How it works screen (optionally with banner)
        lang = _user_lang(update, context)
        await show_how(update, context, lang)
    elif action == "quality":
        lang = _user_lang(update, context)
        # Optional banner first
        _banner = _pick_banner(QUALITY_BANNER_S3_DAY, QUALITY_BANNER_S3_NIGHT, QUALITY_BANNER_S3)
        if _banner:
            try:
                from ..infra.s3 import download_bytes as _dl
                content = _dl(_banner)
                await context.bot.send_photo(chat_id=q.message.chat.id, photo=content)
            except Exception:
                pass
        await show_quality(update, context, lang)
    elif action == "link":
        await link_inline(update, context)
    elif action == "promo":
        lang = _user_lang(update, context)
        await show_promo(update, context, lang)
    elif action == "main":
        try:
            context.user_data["onboarded"] = True
        except Exception:
            pass
        try:
            await start(update, context)
        except Exception:
            pass


async def affiliate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    lang = _user_lang(update, context)
    # Compose status line
    uid = q.from_user.id if q else update.effective_user.id
    status_line = None
    try:
        code, pct = get_affiliate_for_user(uid)
        if code:
            status_line = (f"Вы — партнёр. Код: {code}, {pct}%" if lang=="ru" else f"You are affiliate. Code: {code}, {pct}%")
        elif has_pending_affiliate_request(uid):
            status_line = ("Заявка на рассмотрении" if lang=="ru" else "Request pending")
        else:
            status_line = ("Не являетесь партнёром" if lang=="ru" else "Not an affiliate")
    except Exception:
        pass
    kb = [
        [
            InlineKeyboardButton(_t(lang, "affiliate_become"), callback_data="aff:become"),
            InlineKeyboardButton(_t(lang, "affiliate_stats"), callback_data="aff:stats"),
        ],
        [
            InlineKeyboardButton(_t(lang, "affiliate_list"), callback_data="aff:list"),
            InlineKeyboardButton(_t(lang, "affiliate_dash_btn"), callback_data="aff:dash"),
        ],
        [
            InlineKeyboardButton(_t(lang, "affiliate_payout_btn"), callback_data="aff:payout"),
            InlineKeyboardButton(_t(lang, "affiliate_request"), callback_data="aff:request"),
        ],
    ]
    if EXTERNAL_AFF_LINK_URL:
        kb.append([InlineKeyboardButton(EXTERNAL_AFF_LINK_TEXT, url=EXTERNAL_AFF_LINK_URL)])
    # Admin panel entry
    if _is_admin(uid):
        kb.append([InlineKeyboardButton("🛠️ Admin", callback_data="aff:admin")])
    if q:
        try:
            await q.edit_message_text((status_line + "\n\n" if status_line else "") + _t(lang, "affiliate_menu"), reply_markup=InlineKeyboardMarkup(kb))
        except Exception:
            await q.message.reply_text((status_line + "\n\n" if status_line else "") + _t(lang, "affiliate_menu"), reply_markup=InlineKeyboardMarkup(kb))
    else:
        await update.message.reply_text((status_line + "\n\n" if status_line else "") + _t(lang, "affiliate_menu"), reply_markup=InlineKeyboardMarkup(kb))


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
            deeplink = f"https://t.me/{me.username}?start=ref_{code}"
            share = f"https://t.me/share/url?url={deeplink}"
            text += (f"\n\nShare: {share}" if lang == "en" else f"\n\nПоделиться: {share}")
        except Exception:
            pass
        try:
            await q.edit_message_text(text)
        except Exception:
            await q.message.reply_text(text)
    elif action == "dash":
        # Render 30d dashboard for referrals/commissions
        from io import BytesIO
        import matplotlib.pyplot as plt
        import pandas as pd
        from ..infra.db import get_conn
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT date_trunc('day', created_at)::date AS day,
                               COUNT(1) AS referrals,
                               COALESCE(SUM(commission),0) AS commission
                        FROM referrals
                        WHERE partner_user_id=%s AND created_at >= now() - interval '30 days'
                        GROUP BY 1
                        ORDER BY 1
                        """,
                        (q.from_user.id,),
                    )
                    rows = cur.fetchall() or []
            if not rows:
                try:
                    await q.edit_message_text(_t(lang, "affiliate_dash") + "\n(нет данных)")
                except Exception:
                    await q.message.reply_text(_t(lang, "affiliate_dash") + "\n(нет данных)")
                return
            df = pd.DataFrame(rows, columns=["day", "referrals", "commission"]).set_index("day")
            plt.style.use('seaborn-v0_8')
            fig, ax1 = plt.subplots(figsize=(7.2, 3.6), dpi=200)
            ax2 = ax1.twinx()
            ax1.bar(df.index, df["referrals"], color="#1f77b4", alpha=0.6, label=("Заявки" if lang=="ru" else "Referrals"))
            ax2.plot(df.index, df["commission"], color="#ff7f0e", linewidth=2.0, label=("Начисления" if lang=="ru" else "Accruals"))
            ax1.set_ylabel("#" if lang=="en" else "# заявок")
            ax2.set_ylabel("commission")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            await context.bot.send_photo(chat_id=q.message.chat.id, photo=buf.getvalue(), caption=_t(lang, "affiliate_dash"))
        except Exception as e:
            logger.exception(f"aff dash error: {e}")
            try:
                await q.edit_message_text(_t(lang, "affiliate_dash"))
            except Exception:
                await q.message.reply_text(_t(lang, "affiliate_dash"))
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
        admin_chat = os.getenv("TELEGRAM_ADMIN_CHAT_ID")
        admin_ids = [x.strip() for x in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",") if x.strip()]
        ack = _t(lang, "affiliate_request_ack")
        try:
            req_id = insert_affiliate_request(q.from_user.id, q.from_user.username, None)
            notif_text = (
                f"Affiliate request id={req_id} from @{q.from_user.username or q.from_user.id} "
                f"(id={q.from_user.id}). Approve: /affapprove {q.from_user.id} 50 {req_id} — or mark: /affmark {req_id} approved|rejected"
            )
            if owner:
                await context.bot.send_message(chat_id=owner, text=notif_text)
            if admin_chat:
                await context.bot.send_message(chat_id=admin_chat, text=notif_text)
            for aid in admin_ids:
                try:
                    await context.bot.send_message(chat_id=aid, text=notif_text)
                except Exception:
                    continue
        except Exception:
            pass
        try:
            await q.edit_message_text(ack)
        except Exception:
            await q.message.reply_text(ack)
    elif action == "payout":
        # Create payout request via affiliate_requests and notify owner
        owner = os.getenv("TELEGRAM_OWNER_ID")
        ack = _t(lang, "affiliate_payout_ack")
        try:
            from ..infra.db import insert_affiliate_request, get_conn
            rid = insert_affiliate_request(q.from_user.id, q.from_user.username, note="payout")
            # load balance
            bal = 0
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT balance FROM affiliates WHERE partner_user_id=%s", (q.from_user.id,))
                    row = cur.fetchone()
                    bal = int(row[0] or 0) if row else 0
            if owner:
                await context.bot.send_message(
                    chat_id=owner,
                    text=(
                        f"Payout request id={rid} from @{q.from_user.username or q.from_user.id} "
                        f"(id={q.from_user.id}). Current balance={bal}. Mark: /affmark {rid} approved|rejected"
                    ),
                )
        except Exception:
            pass
        try:
            await q.edit_message_text(ack)
        except Exception:
            await q.message.reply_text(ack)
    elif action == "admin":
        lang = _user_lang(update, context)
        if not _is_admin(q.from_user.id):
            return await q.answer(_t(lang, "not_enough_rights"), show_alert=True)
        kb = [
            [InlineKeyboardButton(_t(lang, "admin_promo"), callback_data="admin:promo")],
            [InlineKeyboardButton(_t(lang, "admin_edit_bonuses"), callback_data="admin:bonuses"), InlineKeyboardButton("📜 Список", callback_data="admin:bonuses_list")],
            [InlineKeyboardButton(_t(lang, "admin_add_news"), callback_data="admin:news"), InlineKeyboardButton("📜 Список", callback_data="admin:news_list")],
            [InlineKeyboardButton(_t(lang, "admin_edit_hero_a"), callback_data="admin:hero:a"), InlineKeyboardButton(_t(lang, "admin_edit_hero_b"), callback_data="admin:hero:b")],
            [InlineKeyboardButton("👥 Affiliates", callback_data="admin:aff")],
        ]
        try:
            await q.edit_message_text(_t(lang, "admin_menu"), reply_markup=InlineKeyboardMarkup(kb))
        except Exception:
            await q.message.reply_text(_t(lang, "admin_menu"), reply_markup=InlineKeyboardMarkup(kb))


async def post_init(app: Application) -> None:
    commands_en = [
        ("start", "start menu"),
        ("buy", "buy subscription"),
        ("status", "subscription status"),
        ("renew", "renew subscription"),
        ("redeem", "redeem code"),
        ("help", "show help"),
        ("ping", "health check"),
        ("id", "show your id"),
    ]
    commands_ru = [
        ("start", "стартовое меню"),
        ("buy", "оплата подписки"),
        ("status", "статус подписки"),
        ("renew", "продление"),
        ("redeem", "активация кода"),
        ("help", "подсказка"),
        ("ping", "проверка связи"),
        ("id", "ваш ID"),
    ]
    await app.bot.set_my_commands(commands_en, language_code="en")
    await app.bot.set_my_commands(commands_ru, language_code="ru")


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    kb = [[InlineKeyboardButton(_t(lang, "lang_choose"), callback_data="settings:lang")]]
    try:
        uid = update.effective_user.id
        if _is_admin(uid):
            kb.append([InlineKeyboardButton("🛠 Admin", callback_data="settings:admin")])
    except Exception:
        pass
    await update.message.reply_text("Settings:", reply_markup=InlineKeyboardMarkup(kb))


async def settings_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.data:
        return
    await q.answer()
    _, action = q.data.split(":", 1)
    if action == "lang":
        return await lang_cmd(update, context)
    if action == "admin":
        # open admin menu
        if not _is_admin(q.from_user.id):
            return await q.answer(_t(_user_lang(update, context), "not_enough_rights"), show_alert=True)
        # reuse admin menu rendering
        try:
            await admin_cb(
                update, context
            )  # admin_cb will look at q.data; ensure menu
        except Exception:
            # fallback explicit
            kb = [
                [InlineKeyboardButton(_t(_user_lang(update, context), "admin_promo"), callback_data="admin:promo")],
                [InlineKeyboardButton(_t(_user_lang(update, context), "admin_edit_bonuses"), callback_data="admin:bonuses"), InlineKeyboardButton("📜 Список", callback_data="admin:bonuses_list")],
                [InlineKeyboardButton(_t(_user_lang(update, context), "admin_add_news"), callback_data="admin:news"), InlineKeyboardButton("📜 Список", callback_data="admin:news_list")],
            ]
            await q.edit_message_text(_t(_user_lang(update, context), "admin_menu"), reply_markup=InlineKeyboardMarkup(kb))
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
    
    async def genpromo(update: Update, context: ContextTypes.DEFAULT_TYPE):
        lang = _user_lang(update, context)
        if not _is_admin(update.message.from_user.id):
            await update.message.reply_text(_t(lang, "not_enough_rights"))
            return
        try:
            months = int(context.args[0]) if context.args else 1
            count = int(context.args[1]) if len(context.args) > 1 else 1
            from ..infra.db import create_redeem_code
            codes: list[str] = []
            for _ in range(max(1, min(count, 50))):
                codes.append(create_redeem_code(months, f"admin_{update.message.from_user.id}"))
            await update.message.reply_text("\n".join([f"/redeem {c}" for c in codes]))
        except Exception as e:
            logger.exception(f"genpromo error: {e}")
            await update.message.reply_text("Error")
    app.add_handler(CommandHandler("genpromo", genpromo))
    
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
    app.add_handler(CommandHandler("settings", lang_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("menu", start))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(CallbackQueryHandler(lang_cb, pattern=r"^lang:(ru|en)$"))
    app.add_handler(CallbackQueryHandler(menu_cb, pattern=r"^menu:(lang|pay|about|affiliate)$"))
    app.add_handler(CallbackQueryHandler(menu_cb, pattern=r"^menu:(insight|link|how|quality)$"))
    app.add_handler(CallbackQueryHandler(intro_start_cb, pattern=r"^intro:start$"))
    # Reply keyboard router must go before generic text handler
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), menu_router))
    app.add_handler(CallbackQueryHandler(plan_cb, pattern=r"^plan:(1|12)$"))
    app.add_handler(
        CallbackQueryHandler(pay_cb, pattern=r"^pay:(1|12):(stars|crypto)$")
    )
    app.add_handler(CallbackQueryHandler(affiliate_cb, pattern=r"^aff:(become|stats)$"))
    app.add_handler(CallbackQueryHandler(affiliate_cb, pattern=r"^aff:(list|request)$"))
    app.add_handler(CallbackQueryHandler(affiliate_cb, pattern=r"^aff:(dash|payout)$"))
    app.add_handler(CallbackQueryHandler(admin_cb, pattern=r"^admin:.*$"))
    app.add_handler(CallbackQueryHandler(settings_cb, pattern=r"^settings:(lang|admin)$"))
    app.add_handler(CallbackQueryHandler(affiliate_cb, pattern=r"^aff:admin$"))
    app.add_handler(CallbackQueryHandler(admin_cb, pattern=r"^admin:(promo|bonuses|news).*$"))
    # successful payment is a Message update
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))
    
    async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Capture free-form text if awaiting_insight is set
        lang = _user_lang(update, context)
        user = update.message.from_user
        txt = update.message.text or ""
        # Admin awaiting flows
        try:
            if context.user_data.get("awaiting_bonuses") and _is_admin(user.id):
                context.user_data["awaiting_bonuses"] = False
                set_content_block("bonus", txt, user.id)
                await update.message.reply_text("Добавлено в бонусы.")
                return
            if context.user_data.get("awaiting_news") and _is_admin(user.id):
                context.user_data["awaiting_news"] = False
                add_news_item(txt, user.id)
                await update.message.reply_text("Новость добавлена.")
                return
            if context.user_data.get("awaiting_hero") and _is_admin(user.id):
                which = context.user_data.get("awaiting_hero")
                context.user_data["awaiting_hero"] = None
                set_content_block(f"hero_{which}", txt, user.id)
                await update.message.reply_text(f"Hero {which} обновлён.")
                return
        except Exception:
            pass
        # Insight flow for subscribers
        try:
            if not context.user_data.get("awaiting_insight"):
                return
            context.user_data["awaiting_insight"] = False
        except Exception:
            return
        st, _ = get_subscription_status(user.id)
        if st != "active":
            await update.message.reply_text(_t(lang, "insight_no_active"))
            return
        try:
            res = evaluate_and_store_insight(user.id, txt)
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
    
    async def handle_admin_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            uid = update.message.from_user.id
            if not _is_admin(uid):
                return
            if not context.user_data.get("awaiting_news"):
                return
            # accept document or photo
            caption = update.message.caption or ""
            content_bytes = None
            kind = None
            if update.message.document:
                file = await update.message.document.get_file()
                content_bytes = await file.download_as_bytearray()
                kind = "DOC"
            elif update.message.photo:
                file = await update.message.photo[-1].get_file()
                content_bytes = await file.download_as_bytearray()
                kind = "PHOTO"
            else:
                return
            from ..infra.s3 import upload_bytes
            key = f"news/{uid}/{int(os.times().elapsed*1000)}"
            s3 = upload_bytes(key, bytes(content_bytes), content_type="application/octet-stream")
            add_news_item(f"{kind}:{s3}|{caption}", uid)
            context.user_data["awaiting_news"] = False
            await update.message.reply_text("Файл добавлен в новости")
        except Exception:
            pass

    app.add_handler(MessageHandler((filters.Document.ALL | filters.PHOTO) & (~filters.COMMAND), handle_admin_files))
    # Unknown commands fallback
    app.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))
    # Error handler
    app.add_error_handler(error_handler)
    logger.info("Payments bot started")
    app.run_polling(allowed_updates=["message", "pre_checkout_query", "callback_query"])


if __name__ == "__main__":
    main()
