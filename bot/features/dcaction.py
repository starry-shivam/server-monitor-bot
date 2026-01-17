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
from pathlib import Path

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes

from bot.features import cb_sign
from bot.config import (
    DOCKER_APPS_DIR,
    DC_SCRIPT,
    DC_ALLOWED_ACTIONS,
    DC_BULK_ACTIONS,
    DC_IGNORE_DIRS,
    CALLBACK_TTL,
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


def is_compose_status(dir_path: Path, status: str) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "compose", "ps", "-q", "--filter", f"status={status}"],
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
    # Validation Checks
    if not dir_path.exists():
        raise FileNotFoundError("Directory does not exist")
    if name in DC_IGNORE_DIRS:
        raise PermissionError(f"`{name}` is ignored permanently")
    if not has_compose_file(dir_path):
        raise RuntimeError("No docker compose file found")

    # Only run status checks for actions that require it
    if action == "up" and is_compose_status(dir_path, "running"):
        raise RuntimeError("Containers are already running")
    elif action == "stop" and not is_compose_status(dir_path, "running"):
        raise RuntimeError("Containers are already stopped")
    elif action == "pause" and is_compose_status(dir_path, "paused"):
        raise RuntimeError("Containers are already paused")
    elif action == "unpause" and not is_compose_status(dir_path, "paused"):
        raise RuntimeError("Containers are not paused")

    outputs: list[str] = []

    # Internal helper to run docker compose commands
    def _run_cmd(args: list[str]) -> None:
        """Internal helper to run docker compose commands."""
        proc = subprocess.run(
            ["docker", "compose"] + args,
            cwd=dir_path,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        outputs.append((proc.stdout or "") + (proc.stderr or ""))

    # Prepare commands based on action
    commands_to_run = []

    if action == "restart":
        commands_to_run = [["down"], ["up", "-d", "--no-build"]]

    elif action == "update":
        # Check if local build files exist
        build_locally = any(
            (dir_path / f).exists() for f in ["Dockerfile", "docker-compose.build.yml"]
        )
        first_step = ["build"] if build_locally else ["pull"]
        commands_to_run = [first_step, ["down"], ["up", "-d", "--no-build"]]

    elif action == "logs":
        commands_to_run = [["logs", "--tail", "100"]]

    else:
        # Standard single-step actions (up, stop, pause, unpause, down)
        args = [action]
        if action == "up":
            args.extend(["-d", "--no-build"])
        commands_to_run = [args]

    # Execute all queued commands
    for args in commands_to_run:
        _run_cmd(args)

    output = "".join(outputs).strip()
    return output or "No output."


def run_bulk_dc(action: str) -> str:
    if action not in DC_BULK_ACTIONS:
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
        timeout=2200,
    )

    output = (proc.stdout or "") + (proc.stderr or "")
    return output.strip() or "No output."


# ============== Callback Data ================


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
    sig = cb_sign(payload)
    return f"{payload}:{sig}"


def dc_keyboard(
    user_id: int, ts: int, action: str, target: str = None
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
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


# ================= Handlers =================


def dcaction_help() -> str:
    return (
        "🐋 <b>Docker Compose Action Command</b>\n\n"
        "Use <code>/dcaction</code> to manage your Docker Compose applications.\n\n"
        "<b>Available Commands:</b>\n"
        "‣ <code>/dcaction list</code>\n"
        "‣ <code>/dcaction pull &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction build &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction up &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction up --all</code>\n"
        "‣ <code>/dcaction stop &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction stop --all</code>\n"
        # "‣ <code>/dcaction pause &lt;dir&gt;</code>\n"
        # "‣ <code>/dcaction pause --all</code>\n"
        # "‣ <code>/dcaction unpause &lt;dir&gt;</code>\n"
        # "‣ <code>/dcaction unpause --all</code>\n"
        "‣ <code>/dcaction down &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction update &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction logs &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction restart &lt;dir&gt;</code>"
        "\n\n"
        "<b>Note:</b> Replace <code>&lt;dir&gt;</code> with the name of your "
        "docker app directory as listed in <code>/dcaction list</code>."
    )


@restricted
async def dcaction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args

    if not DOCKER_APPS_DIR.exists():
        return await update.message.reply_text(
            "❌ Docker apps directory is not configured."
        )

    if not args:
        return await update.message.reply_text(
            "❌ Missing arguments.\n" "Use <code>/dcaction help</code> for usage info.",
            parse_mode="HTML",
        )

    # Handle help command first as it is not a real action
    if args[0] == "help":
        return await update.message.reply_text(
            dcaction_help(),
            parse_mode="HTML",
        )

    # Handle list command separately as it is not a real action
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

    # validate action
    action = args[0].lower()
    if action not in DC_ALLOWED_ACTIONS:
        return await update.message.reply_text(
            f"❌ Supported actions: {', '.join(DC_ALLOWED_ACTIONS)}."
        )

    # validate if --all is supported for this action
    is_all = "--all" in args

    if is_all and action not in DC_BULK_ACTIONS:
        return await update.message.reply_text(
            f"❌ The <code>--all</code> option is not supported for "
            f"<code>{action}</code>.\n\n"
            f"Allowed with: {', '.join(DC_BULK_ACTIONS)}",
            parse_mode="HTML",
        )

    # show error if both --all and directory specified
    if is_all and len(args) > 2:
        return await update.message.reply_text(
            "❌ When using <code>--all</code>, do not specify a directory.",
            parse_mode="HTML",
        )

    target = None if is_all else (args[1] if len(args) > 1 else None)

    # single target validation
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
    keyboard = dc_keyboard(user_id, ts, action, target)

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
        return await q.answer("🚫 Invalid callback", show_alert=True)

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
    expected_sig = cb_sign(payload)

    if not hmac.compare_digest(sig, expected_sig):
        return await q.answer(
            "🚫 Invalid signature. Action aborted.",
            show_alert=True,
        )

    await q.answer()  # acknowledge callback

    if cb_type == "cancel":
        return await q.edit_message_text("❌ Cancelled.")

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
