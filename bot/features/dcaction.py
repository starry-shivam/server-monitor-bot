import time
import uuid
import json
import hmac
import hashlib
import base64
import subprocess
from pathlib import Path

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes

from bot.config import (
    DOCKER_APPS_DIR,
    DC_SCRIPT,
    DC_ALLOWED_ACTIONS,
    DC_IGNORE_DIRS,
    CALLBACK_TTL,
    CALLBACK_SIG_SECRET,
)
from bot.auth import restricted


# ================= Docker Compose Utils =================


COMPOSE_FILES = {
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
}


def has_compose_file(dir_path: Path) -> bool:
    return any((dir_path / f).exists() for f in COMPOSE_FILES)


def list_docker_dirs() -> list[str]:
    if not DOCKER_APPS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in DOCKER_APPS_DIR.iterdir()
        if d.is_dir() and d.name not in DC_IGNORE_DIRS
    )


def is_compose_running(dir_path: Path) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "compose", "ps", "-q", "--filter", "status=running"],
            cwd=dir_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return False


def run_single_dc(action: str, name: str) -> str:
    if action not in DC_ALLOWED_ACTIONS:
        raise ValueError("Unsupported action")

    dir_path = DOCKER_APPS_DIR / name

    if not dir_path.exists():
        raise FileNotFoundError("Directory does not exist")

    if name in DC_IGNORE_DIRS:
        raise PermissionError(f"`{name}` is ignored permanently")

    if not has_compose_file(dir_path):
        raise RuntimeError("No docker compose file found")

    running = is_compose_running(dir_path)

    if action == "up" and running:
        raise RuntimeError("Containers are already running")

    if action == "stop" and not running:
        raise RuntimeError("Containers are already stopped")

    outputs: list[str] = []

    def run_cmd(cmd: list[str]) -> None:
        proc = subprocess.run(
            cmd,
            cwd=dir_path,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        outputs.append((proc.stdout or "") + (proc.stderr or ""))

    if action == "restart":
        run_cmd(["docker", "compose", "down"])
        run_cmd(["docker", "compose", "up", "-d"])
    else:
        cmd = ["docker", "compose", action]
        if action == "up":
            cmd.append("-d")
        run_cmd(cmd)

    output = "".join(outputs).strip()
    return output or "No output."


def run_bulk_dc(action: str) -> str:
    if action not in DC_ALLOWED_ACTIONS:
        raise ValueError("Unsupported action")

    cmd = ["bash", DC_SCRIPT, action, "--no-color"]
    if DC_IGNORE_DIRS:
        ignore_arg = ",".join(DC_IGNORE_DIRS)
        cmd.extend(["--ignore", ignore_arg])

    proc = subprocess.run(
        cmd,
        cwd=DOCKER_APPS_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )

    output = (proc.stdout or "") + (proc.stderr or "")
    return output.strip() or "No output."


# ================= Callback Signing =================


def dc_sign(payload: str) -> str:
    sig = hmac.new(
        CALLBACK_SIG_SECRET.encode(),
        payload.encode(),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(sig[:9]).decode().rstrip("=")


def dc_callback_data(
    cb_type: str,
    user_id: int,
    ts: int,
    action: str | None = None,
    target: str | None = None,
) -> str:
    parts = [
        "dc",
        cb_type,
        str(user_id),
        str(ts),
        action or "-",
        target or "-",
    ]
    payload = ":".join(parts)
    sig = dc_sign(payload)
    return f"{payload}:{sig}"


# ================= Handlers =================


@restricted
async def dcaction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args

    if not args:
        return await update.message.reply_text(
            "❌ <b>Usage:</b>\n"
            "‣ <code>/dcaction list</code>\n"
            "‣ <code>/dcaction pull &lt;dir&gt;</code>\n"
            "‣ <code>/dcaction build &lt;dir&gt;</code>\n"
            "‣ <code>/dcaction up &lt;dir&gt;</code>\n"
            "‣ <code>/dcaction up --all</code>\n"
            "‣ <code>/dcaction stop &lt;dir&gt;</code>\n"
            "‣ <code>/dcaction stop --all</code>\n"
            "‣ <code>/dcaction restart &lt;dir&gt;</code>",
            parse_mode="HTML",
        )

    if args[0] == "list":
        dirs = list_docker_dirs()
        if not dirs:
            return await update.message.reply_text(
                "❌ No docker app directories found."
            )

        text = "📦 <b>Available Docker Apps:</b>\n\n"
        for i, d in enumerate(dirs, 1):
            text += f"{i}. <code>{d}</code>\n"

        return await update.message.reply_text(text, parse_mode="HTML")

    action = args[0].lower()
    if action not in DC_ALLOWED_ACTIONS:
        return await update.message.reply_text(
            f"❌ Supported actions: {', '.join(DC_ALLOWED_ACTIONS)}."
        )

    is_all = "--all" in args
    target = None if is_all else (args[1] if len(args) > 1 else None)

    if not is_all:
        if not target:
            return await update.message.reply_text("❌ Missing directory name.")

        if target in DC_IGNORE_DIRS:
            return await update.message.reply_text(
                f"🚫 `{target}` is in ignore list.",
                parse_mode="Markdown",
            )

        dir_path = DOCKER_APPS_DIR / target
        if not dir_path.exists():
            return await update.message.reply_text(
                f"❌ Directory `{target}` not found.\n"
                f"Run `/dcaction list` to see available apps.",
                parse_mode="Markdown",
            )

        if not has_compose_file(dir_path):
            return await update.message.reply_text(
                f"❌ `{target}` has no docker compose file.",
                parse_mode="Markdown",
            )

    user_id = update.effective_user.id
    ts = int(time.time())

    preview = (
        f"⚠️ <b>Confirm Docker Action</b>\n\n"
        f"• Action: <code>{action}</code>\n"
        f"• Target: <code>{target or 'ALL'}</code>\n\n"
        "Proceed?"
    )

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Confirm",
                    callback_data=dc_callback_data(
                        "run",
                        user_id,
                        ts,
                        action,
                        target or "ALL",
                    ),
                ),
                InlineKeyboardButton(
                    "❌ Cancel",
                    callback_data=dc_callback_data(
                        "cancel",
                        user_id,
                        ts,
                    ),
                ),
            ]
        ]
    )

    await update.message.reply_text(
        preview,
        parse_mode="HTML",
        reply_markup=keyboard,
    )


async def dcaction_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    user = q.from_user

    parts = q.data.split(":")
    if len(parts) != 7 or parts[0] != "dc":
        return

    _, cb_type, uid, ts, action, target, sig = parts

    uid = int(uid)
    ts = int(ts)
    now = int(time.time())

    # Owner check
    if user.id != uid:
        return await q.answer("🚫 Unauthorized", show_alert=True)

    # Expiry check
    if now - ts > CALLBACK_TTL:
        return await q.answer(
            "⏱ This action has expired. Please run the command again.",
            show_alert=True,
        )

    # Signature check
    payload = ":".join(parts[:-1])
    expected_sig = dc_sign(payload)

    if not hmac.compare_digest(sig, expected_sig):
        return await q.answer(
            "🚨 Invalid or tampered callback.",
            show_alert=True,
        )

    await q.answer()  # Acknowledge callback

    if cb_type == "cancel":
        return await q.edit_message_text("❌ Cancelled.")

    # Execute action
    await q.edit_message_text("🐋 Executing…")

    try:
        if target == "ALL":
            output = run_bulk_dc(action)
        else:
            output = run_single_dc(action, target)

        if len(output) > 4000:
            output = output[-4000:]

        await q.edit_message_text(
            f"📊 <b>Execution Summary</b>\n\n<pre>{output}</pre>",
            parse_mode="HTML",
        )

    except Exception as e:
        await q.edit_message_text(
            f"❌ Error:\n<code>{e}</code>",
            parse_mode="HTML",
        )
