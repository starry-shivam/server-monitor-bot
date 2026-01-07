import shlex
import subprocess
import time
import hmac
import hashlib
import base64
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
    command: str | None = None,
) -> str:
    parts = [
        "sh",
        cb_type,
        str(user_id),
        str(ts),
        command or "-",
    ]
    payload = ":".join(parts)
    sig = shell_sign(payload)
    return f"{payload}:{sig}"


# ================= Handler =================


@restricted
async def shell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        cmd = update.message.text.split(None, 1)[1]
    except IndexError:
        return await update.message.reply_text("❌ No command provided.")

    # && is not supported by subprocess, except in a shell context
    # i.e. subprocess.run("cmd1 && cmd2", shell=True), which is unsafe.
    if "&&" in cmd or ";" in cmd:
        return await update.message.reply_text(
            "🚫 Multiple commands are not allowed. " "Please run one command at a time."
        )

    user_id = update.effective_user.id
    ts = int(time.time())

    preview = (
        "⚠️ <b>Confirm Shell Command</b>\n\n" f"<pre>{escape(cmd)}</pre>\n\n" "Proceed?"
    )

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Confirm",
                    callback_data=shell_callback_data("run", user_id, ts, cmd),
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


# ================= Callback =================


async def shell_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    user = q.from_user

    parts = q.data.split(":")
    if len(parts) != 6 or parts[0] != "sh":
        return

    _, cb_type, uid, ts, command, sig = parts
    uid = int(uid)
    ts = int(ts)
    now = int(time.time())

    # Owner validation
    if user.id != uid:
        return await q.answer("🚫 Unauthorized", show_alert=True)

    # Expiry check
    if now - ts > CALLBACK_TTL:
        return await q.answer(
            "⏱ This action has expired.",
            show_alert=True,
        )

    payload = ":".join(parts[:-1])
    expected_sig = shell_sign(payload)

    if not hmac.compare_digest(sig, expected_sig):
        return await q.answer(
            "🚨 Invalid or tampered callback.",
            show_alert=True,
        )

    await q.answer()  # Acknowledge the callback to avoid "loading" state

    if cb_type == "cancel":
        return await q.edit_message_text("❌ Cancelled.")

    # ▶ Execute
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
