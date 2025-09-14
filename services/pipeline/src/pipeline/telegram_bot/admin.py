from __future__ import annotations
import os

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from ..infra.db import (
    create_redeem_code,
    list_content_items_with_id,
    delete_content_item,
    add_news_item,
)
from ..trading.subscriptions import sweep_and_revoke_channel_access
from ..infra.s3 import upload_bytes
from .menus import _t, _user_lang, _is_admin

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
        item_id = data.replace("admin:bonus_del:", "")
        try:
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
        item_id = data.replace("admin:news_del:", "")
        try:
            delete_content_item(item_id)
            await q.edit_message_text("Удалено.")
        except Exception:
            await q.edit_message_text("Ошибка удаления.")
        return

async def admin_sweep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    if not _is_admin(update.message.from_user.id):
        await update.message.reply_text(_t(lang, "not_enough_rights"))
        return
    count = sweep_and_revoke_channel_access()
    await update.message.reply_text(_t(lang, "sweep_done", count=count))

async def genpromo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _user_lang(update, context)
    if not _is_admin(update.message.from_user.id):
        await update.message.reply_text(_t(lang, "not_enough_rights"))
        return
    try:
        months = int(context.args[0]) if context.args else 1
        count = int(context.args[1]) if len(context.args) > 1 else 1
        codes = [create_redeem_code(months, f"admin_{update.message.from_user.id}") for _ in range(max(1, min(count, 50)))]
        await update.message.reply_text("\n".join([f"/redeem {c}" for c in codes]))
    except Exception as e:
        logger.exception(f"genpromo error: {e}")
        await update.message.reply_text("Error")

async def handle_admin_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        uid = update.message.from_user.id
        if not _is_admin(uid):
            return
        if not context.user_data.get("awaiting_news"):
            return
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
        key = f"news/{uid}/{int(os.times().elapsed*1000)}"
        s3 = upload_bytes(key, bytes(content_bytes), content_type="application/octet-stream")
        add_news_item(f"{kind}:{s3}|{caption}", uid)
        context.user_data["awaiting_news"] = False
        await update.message.reply_text("Файл добавлен в новости")
    except Exception:
        pass

