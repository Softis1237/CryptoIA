# flake8: noqa
from typing import Dict

from pydantic import BaseModel


class MessageCatalog(BaseModel):
    ru: Dict[str, str]
    en: Dict[str, str]


MESSAGES = MessageCatalog(
    ru={
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
        "external_aff_link": "💠 Бонусы на бирже",
        "admin_menu": "Панель администратора:",
        "admin_promo": "🎁 Сгенерировать коды",
        "admin_edit_bonuses": "✍️ Редактировать бонусы",
        "admin_add_news": "📝 Добавить новость",
        "admin_edit_hero_a": "🖊 Hero A",
        "admin_edit_hero_b": "🖊 Hero B",
        "admin_hint_bonus": "Отправьте новый пункт бонусов (каждый пункт отдельным сообщением)",
        "admin_hint_news": "Добавить новость?",
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
        "affiliate_dash_btn": "📈 Дашборд",
        "affiliate_payout_btn": "💸 Запросить выплату",
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
    en={
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
        "external_aff_link": "💠 Exchange bonus",
        "admin_menu": "Admin panel:",
        "admin_promo": "🎁 Generate codes",
        "admin_edit_bonuses": "✍️ Edit bonuses",
        "admin_add_news": "📝 Add news",
        "admin_edit_hero_a": "🖊 Hero A",
        "admin_edit_hero_b": "🖊 Hero B",
        "admin_hint_bonus": "Send a new bonus item",
        "admin_hint_news": "Add news?",
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
        "affiliate_dash_btn": "📈 Dashboard",
        "affiliate_payout_btn": "💸 Request payout",
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
)

_MESSAGES = MESSAGES.model_dump()


def _t(lang_code: str, key: str, **kwargs) -> str:
    return (_MESSAGES.get(lang_code, _MESSAGES["ru"]).get(key, key)).format(**kwargs)
