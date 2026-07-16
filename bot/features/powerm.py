#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025-Present Stɑrry Shivɑm <starry@krsh.dev>
# All Rights Reserved. // This file is a part of server-monitor-bot
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import hmac
import subprocess
import asyncio
import logging

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler

from bot.features import cb_sign
from bot.auth import restricted, is_authorized_callback_user
from bot.config import CALLBACK_TTL, POWER_MGMT_AVAILABLE
from bot.logger import log_callback, log_security_event

log = logging.getLogger(__name__)

# ================= Configuration =================

SHUTDOWN_COUNTDOWN = 5  # seconds

POWER_ACTIONS = {
    "reboot": {
        "title": "🔁 <b>Reboot Server</b>",
        "summary": (
            "⚠️ <b>Reboot Server</b>\n\n"
            "‣ All services will restart\n"
            "‣ Active connections will drop\n\n"
            "Proceed?"
        ),
        "command": "reboot",
    },
    "poweroff": {
        "title": "⚠️ <b>Power Off Server</b>",
        "summary": (
            "⚠️ <b>Power Off Server</b>\n\n"
            "‣ All services will stop\n"
            "‣ Manual startup will be required\n\n"
            "Proceed?"
        ),
        "command": "poweroff",
    },
}

POWER_HELPER_BIN = "/usr/local/bin/power-helper"


# ================= Callback Data =================


def _cb(action: str, user_id: int, ts: int, phase: str) -> str:
    payload = f"pw:{phase}:{action}:{user_id}:{ts}"
    return f"{payload}:{cb_sign(payload)}"


# ================= Commands ====+================
@restricted
async def reboot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _prompt(update, context, "reboot")


@restricted
async def poweroff(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _prompt(update, context, "poweroff")


async def _prompt(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    action: str,
):
    meta = POWER_ACTIONS[action]
    user_id = update.effective_user.id
    ts = int(time.time())

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Confirm",
                    callback_data=_cb(action, user_id, ts, "confirm"),
                ),
                InlineKeyboardButton(
                    "❌ Cancel",
                    callback_data=_cb(action, user_id, ts, "cancel"),
                ),
            ]
        ]
    )

    await update.message.reply_text(
        meta["summary"],
        parse_mode="HTML",
        reply_markup=keyboard,
    )


# ================= Callback Handler =================


async def power_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    user = q.from_user

    parts = q.data.split(":")
    if len(parts) != 6 or parts[0] != "pw":
        return await q.answer("🚫 Invalid callback.", show_alert=True)

    _, phase, action, uid, ts, sig = parts
    uid = int(uid)
    ts = int(ts)
    now = int(time.time())

    # Owner check
    if not is_authorized_callback_user(getattr(user, "id", None), uid):
        log_security_event(
            log,
            "power_callback",
            "blocked",
            detail=f"uid={getattr(user, 'id', '?')} expected={uid}",
        )
        return await q.answer("🚫 Unauthorized", show_alert=True)

    # TTL check
    if now - ts > CALLBACK_TTL:
        return await q.answer("⏱ Action expired.", show_alert=True)

    # Signature check
    payload = ":".join(parts[:-1])
    if not hmac.compare_digest(sig, cb_sign(payload)):
        log_security_event(log, "power_callback", "invalid_signature")
        return await q.answer("🚫 Invalid signature.", show_alert=True)

    await q.answer()  # Acknowledge callback to remove loading state

    # Cancel
    if phase == "cancel":
        log_callback(log, user, "power", action, "cancelled")
        task = context.bot_data.pop("shutdown_task", None)
        if task:
            task.cancel()
        return await q.edit_message_text("❌ Cancelled.")

    if phase != "confirm" or action not in POWER_ACTIONS:
        return

    # Prevent multiple countdowns
    if context.bot_data.get("shutdown_task"):
        log_callback(log, user, "power", action, "already_running")
        return await q.edit_message_text("⚠️ Shutdown already in progress.")

    meta = POWER_ACTIONS[action]

    async def countdown():
        try:
            for i in range(SHUTDOWN_COUNTDOWN, 0, -1):
                await q.edit_message_text(
                    f"{meta['title']}\n\n" f"Shutting down in <b>{i}</b>…",
                    parse_mode="HTML",
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "❌ Cancel",
                                    callback_data=_cb(action, uid, ts, "cancel"),
                                )
                            ]
                        ]
                    ),
                )
                await asyncio.sleep(1)

            # Delivery-safe final message
            await q.edit_message_text(
                "✅ <b>Action confirmed</b>\n\n" "Server is shutting down now.",
                parse_mode="HTML",
            )

            # Give Telegram time to deliver
            await asyncio.sleep(1)

            log_callback(
                log,
                user,
                "power",
                action,
                "executed",
                detail=f"helper={POWER_HELPER_BIN} action={meta['command']}",
            )

            subprocess.run(
                [POWER_HELPER_BIN, meta["command"]],
                check=True,
                timeout=5,
            )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log_callback(log, user, "power", action, "failed", detail=str(e))

    task = asyncio.create_task(countdown())
    context.bot_data["shutdown_task"] = task


def get_help_section() -> str | None:
    if not POWER_MGMT_AVAILABLE:
        return None
    return (
        "‣ <code>/reboot</code> — Reboot the server\n"
        "‣ <code>/poweroff</code> — Power off the server"
    )


def get_commands() -> list[tuple[str, str]]:
    if not POWER_MGMT_AVAILABLE:
        return []
    return [
        ("reboot", "Reboot the server"),
        ("poweroff", "Power off the server"),
    ]


def register_handlers(app):
    if not POWER_MGMT_AVAILABLE:
        return

    app.add_handler(CommandHandler("reboot", reboot))
    app.add_handler(CommandHandler("poweroff", poweroff))
    app.add_handler(CallbackQueryHandler(power_callback, pattern=r"^pw:"))
