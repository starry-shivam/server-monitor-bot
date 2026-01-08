import shlex
import subprocess
import time
import hmac
import hashlib
import base64
import secrets
from enum import Enum
from html import escape

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes

from bot.config import (
    SHELL_DENYLIST,
    SHELL_TIMEOUT,
    SHELL_MAX_OUTPUT,
    CALLBACK_SIG_SECRET,
    CALLBACK_TTL,
)
from bot.auth import restricted


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

    parts = shlex.split(command)
    if any(p in SHELL_DENYLIST for p in parts):
        raise PermissionError("Blocked command")

    proc = subprocess.run(
        parts,
        capture_output=True,
        text=True,
        timeout=SHELL_TIMEOUT,
    )

    output = (proc.stdout or "") + (proc.stderr or "")
    return output[-SHELL_MAX_OUTPUT:] or "No output."


# ================= Callback Signing =================


def shell_sign(payload: str) -> str:
    sig = hmac.new(
        CALLBACK_SIG_SECRET.encode(),
        payload.encode(),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(sig[:9]).decode().rstrip("=")


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
    sig = shell_sign(payload)
    return f"{payload}:{sig}"


# ================= /shell Command =================


@restricted
async def shell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        cmd = update.message.text.split(None, 1)[1]
    except IndexError:
        return await update.message.reply_text("❌ No command provided.")

    if "&&" in cmd or ";" in cmd:
        return await update.message.reply_text("🚫 Multiple commands are not allowed.")

    user_id = update.effective_user.id
    ts = int(time.time())
    cmd_id = secrets.token_urlsafe(8)

    store_command(cmd_id, cmd)

    preview = (
        "⚠️ <b>Confirm Shell Command</b>\n\n" f"<pre>{escape(cmd)}</pre>\n\n" "Proceed?"
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
        return

    _, cb_type, uid, ts, cmd_id, sig = parts
    uid = int(uid)
    ts = int(ts)

    # Owner check
    if user.id != uid:
        return await q.answer("🚫 Unauthorized", show_alert=True)

    # Signature check
    payload = ":".join(parts[:-1])
    if not hmac.compare_digest(sig, shell_sign(payload)):
        return await q.answer(
            "🚫 Invalid signature. Action aborted.",
            show_alert=True,
        )

    await q.answer()

    if cb_type == "cancel":
        return await q.edit_message_text("❌ Cancelled.")

    status, command = pop_command(cmd_id, CALLBACK_TTL)

    if status == CmdStatus.EXPIRED:
        return await q.edit_message_text("⏱ This command has expired.")

    if status == CmdStatus.NOT_FOUND:
        return await q.edit_message_text("🚫 This action is no longer valid.")

    await q.edit_message_text("💻 Executing…")

    try:
        output = _shell_exec(command)
        await q.edit_message_text(
            f"<pre>{escape(output)}</pre>",
            parse_mode="HTML",
        )
    except PermissionError as e:
        await q.edit_message_text(
            f"🚫 {escape(str(e))}",
            parse_mode="HTML",
        )
    except subprocess.TimeoutExpired:
        await q.edit_message_text("⏱ Command timed out.")
    except Exception as e:
        await q.edit_message_text(
            f"❌ {escape(str(e))}",
            parse_mode="HTML",
        )
