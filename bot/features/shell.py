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

import shlex
import subprocess
import time
import hmac
import secrets
import logging
from enum import Enum
from html import escape

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler

from bot.features import cb_sign
from bot.logger import log_callback, log_security_event
from bot.config import (
    SHELL_ALLOWED_COMMANDS,
    SHELL_FORBIDDEN_CHARS,
    SHELL_TIMEOUT,
    SHELL_MAX_OUTPUT,
    CALLBACK_TTL,
)
from bot.auth import restricted, is_authorized_callback_user

log = logging.getLogger(__name__)

# ================= In-Memory Command Store =================

_SHELL_PENDING: dict[str, dict] = {}


class CmdStatus(str, Enum):
    OK = "ok"
    EXPIRED = "expired"
    NOT_FOUND = "not_found"


def store_command(cmd_id: str, command: str):
    _SHELL_PENDING[cmd_id] = {
        "command": command,
        "ts": int(time.time()),
    }


def pop_command(cmd_id: str, ttl: int) -> tuple[CmdStatus, str | None]:
    entry = _SHELL_PENDING.get(cmd_id)

    if not entry:
        return CmdStatus.NOT_FOUND, None

    now = int(time.time())
    if now - entry["ts"] > ttl:
        _SHELL_PENDING.pop(cmd_id, None)
        return CmdStatus.EXPIRED, None

    _SHELL_PENDING.pop(cmd_id, None)
    return CmdStatus.OK, entry["command"]


# ================= Shell Utils =================


def _shell_exec(command: str) -> str:
    command = command.strip()
    if not command:
        raise ValueError("Empty command")

    if any(ch in command for ch in SHELL_FORBIDDEN_CHARS):
        log_security_event(log, "shell_command", "blocked", detail=command)
        raise PermissionError("Blocked command")

    parts = shlex.split(command)
    if not parts:
        raise ValueError("Empty command")

    command_name = parts[0]
    if command_name not in SHELL_ALLOWED_COMMANDS:
        log_security_event(log, "shell_command", "blocked", detail=command)
        raise PermissionError("Only approved read-only commands are allowed")

    proc = subprocess.run(
        parts,
        capture_output=True,
        text=True,
        timeout=SHELL_TIMEOUT,
    )

    output = (proc.stdout or "") + (proc.stderr or "")
    return output[-SHELL_MAX_OUTPUT:] or "No output."


# ================= Callback Data =================


def shell_callback_data(
    cb_type: str,
    user_id: int,
    ts: int,
    cmd_id: str | None = None,
) -> str:
    parts = [
        "sh",
        cb_type,
        str(user_id),
        str(ts),
        cmd_id or "-",
    ]
    payload = ":".join(parts)
    sig = cb_sign(payload)
    return f"{payload}:{sig}"


# ================= /shell Command =================


@restricted
async def shell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        cmd = update.message.text.split(None, 1)[1]
    except IndexError:
        return await update.message.reply_text("❌ No command provided.")

    if any(ch in cmd for ch in SHELL_FORBIDDEN_CHARS):
        return await update.message.reply_text(
            "🚫 Only single read-only commands are allowed."
        )

    user_id = update.effective_user.id
    ts = int(time.time())
    cmd_id = secrets.token_urlsafe(8)

    store_command(cmd_id, cmd)
    log_callback(log, update.effective_user, "shell", "queue", "accepted", detail=cmd)

    preview = (
        "⚠️ <b>Confirm Shell Command</b>\n\n"
        f"<pre>{escape(cmd)}</pre>\n\n"
        "For security reasons, only approved read-only commands are allowed. Proceed?"
    )

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Confirm",
                    callback_data=shell_callback_data("run", user_id, ts, cmd_id),
                ),
                InlineKeyboardButton(
                    "❌ Cancel",
                    callback_data=shell_callback_data("cancel", user_id, ts),
                ),
            ]
        ]
    )

    await update.message.reply_text(
        preview,
        parse_mode="HTML",
        reply_markup=keyboard,
    )


# ================= Callback Handler =================


async def shell_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    user = q.from_user

    parts = q.data.split(":")
    if len(parts) != 6 or parts[0] != "sh":
        return await q.answer("🚫 Invalid callback.", show_alert=True)

    _, cb_type, uid, ts, cmd_id, sig = parts
    uid = int(uid)
    ts = int(ts)

    # Owner check
    if not is_authorized_callback_user(getattr(user, "id", None), uid):
        log_security_event(
            log,
            "shell_callback",
            "blocked",
            detail=f"uid={getattr(user, 'id', '?')} expected={uid}",
        )
        return await q.answer("🚫 Unauthorized", show_alert=True)

    # Signature check
    payload = ":".join(parts[:-1])
    if not hmac.compare_digest(sig, cb_sign(payload)):
        log_security_event(log, "shell_callback", "invalid_signature")
        return await q.answer(
            "🚫 Invalid signature. Action aborted.",
            show_alert=True,
        )

    await q.answer()

    if cb_type == "cancel":
        log_callback(log, user, "shell", "cancel", "cancelled")
        return await q.edit_message_text("❌ Cancelled.")

    status, command = pop_command(cmd_id, CALLBACK_TTL)

    if status == CmdStatus.EXPIRED:
        log_callback(log, user, "shell", "run", "expired")
        return await q.edit_message_text("⏱ This command has expired.")

    if status == CmdStatus.NOT_FOUND:
        log_callback(log, user, "shell", "run", "missing")
        return await q.edit_message_text("🚫 This action is no longer valid.")

    await q.edit_message_text("💻 Executing…")

    try:
        output = _shell_exec(command)
        log_callback(log, user, "shell", "run", "executed", detail=command)
        await q.edit_message_text(
            f"<pre>{escape(output)}</pre>",
            parse_mode="HTML",
        )
    except PermissionError as e:
        log_security_event(log, "shell_command", "blocked", detail=command)
        await q.edit_message_text(
            f"🚫 {escape(str(e))}",
            parse_mode="HTML",
        )
    except subprocess.TimeoutExpired:
        log_callback(log, user, "shell", "run", "timeout", detail=command)
        await q.edit_message_text("⏱ Command timed out.")
    except Exception as e:
        log_callback(log, user, "shell", "run", "failed", detail=str(e))
        await q.edit_message_text(
            f"❌ {escape(str(e))}",
            parse_mode="HTML",
        )


def get_help_section() -> str:
    return "‣ <code>/shell</code> — Execute approved read-only shell commands"


def get_commands() -> list[tuple[str, str]]:
    return [("shell", "Execute approved read-only shell commands")]


def register_handlers(app):
    app.add_handler(CommandHandler("shell", shell))
    app.add_handler(CallbackQueryHandler(shell_callback, pattern=r"^sh:"))
