from __future__ import annotations

import os
from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from ..infra.db import (
    get_or_create_affiliate,
    get_affiliate_stats,
    set_affiliate_percent,
    list_referrals,
    insert_affiliate_request,
    mark_affiliate_request,
    list_affiliate_requests,
    get_affiliate_for_user,
    has_pending_affiliate_request,
    get_conn,
)
from .menus import _t, _user_lang, _is_admin, EXTERNAL_AFF_LINK_URL, EXTERNAL_AFF_LINK_TEXT

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

async def affiliate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    lang = _user_lang(update, context)
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
        [InlineKeyboardButton(_t(lang, "affiliate_become"), callback_data="aff:become")],
        [InlineKeyboardButton(_t(lang, "affiliate_dash_btn"), callback_data="aff:dash")],
        [InlineKeyboardButton(_t(lang, "affiliate_stats"), callback_data="aff:stats")],
        [InlineKeyboardButton(_t(lang, "affiliate_list"), callback_data="aff:list")],
        [InlineKeyboardButton(_t(lang, "affiliate_payout_btn"), callback_data="aff:payout")],
        [InlineKeyboardButton(_t(lang, "affiliate_request"), callback_data="aff:request")],
    ]
    if EXTERNAL_AFF_LINK_URL:
        kb.append([InlineKeyboardButton(EXTERNAL_AFF_LINK_TEXT, url=EXTERNAL_AFF_LINK_URL)])
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
        from io import BytesIO
        import matplotlib.pyplot as plt
        import pandas as pd
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
            ax1.bar(df.index, df["referrals"], color="#1f77b4", alpha=0.6)
            ax2.plot(df.index, df["commission"], color="#ff7f0e", linewidth=2.0)
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
                f"• {r[0]} — amt={r[2]} comm={r[3]} at {r[4]}" for r in rows
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
        owner = os.getenv("TELEGRAM_OWNER_ID")
        ack = _t(lang, "affiliate_payout_ack")
        try:
            rid = insert_affiliate_request(q.from_user.id, q.from_user.username, note="payout")
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

