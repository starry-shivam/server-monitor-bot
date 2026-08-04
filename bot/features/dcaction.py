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

import html
import time
import hmac
import shutil
import subprocess
import logging
from pathlib import Path

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler

from bot.features import cb_sign
from bot.logger import log_callback, log_security_event
from bot.config import (
    DOCKER_APPS_DIR,
    DC_ALLOWED_ACTIONS,
    DC_IGNORE_DIRS,
    CALLBACK_TTL,
)
from bot.auth import restricted, is_authorized_callback_user

log = logging.getLogger(__name__)

# ================= Docker Compose Utils =================


COMPOSE_FILES = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
)

BULK_ACTIONS = {"up", "stop", "update"}


def has_compose_file(dir_path: Path) -> bool:
    return any((dir_path / f).exists() for f in COMPOSE_FILES)


def get_compose_file(dir_path: Path) -> Path:
    for file_name in COMPOSE_FILES:
        compose_path = dir_path / file_name
        if compose_path.exists():
            return compose_path
    raise RuntimeError("No docker compose file found")


def is_locally_built(target: str | Path) -> bool:
    dir_path = target if isinstance(target, Path) else DOCKER_APPS_DIR / target
    if not dir_path.exists() or not has_compose_file(dir_path):
        return False
    return any(
        (dir_path / f).exists() for f in ["Dockerfile", "docker-compose.build.yml"]
    )


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
        # special case to check if any containers exist
        # used for "down" action validation.
        if status == "stack_exists":
            command = ["docker", "compose", "ps", "-aq"]
        else:
            allowed = {"running", "paused", "exited"}
            if status not in allowed:
                raise ValueError(f"Status check failed: unsupported status '{status}'")
            command = ["docker", "compose", "ps", "-q", "--filter", f"status={status}"]
        proc = subprocess.run(
            command,
            cwd=dir_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return False


def run_single_dc(action: str, name: str) -> str:
    supported_actions = set(DC_ALLOWED_ACTIONS) | {"config_resolve"}
    if action not in supported_actions:
        raise ValueError("Unsupported action")

    dir_path = DOCKER_APPS_DIR / name
    # Validation Checks
    if not dir_path.exists():
        raise FileNotFoundError("Directory does not exist")
    if name in DC_IGNORE_DIRS:
        raise PermissionError(f"`{name}` is ignored permanently")
    if not has_compose_file(dir_path):
        raise RuntimeError("No docker compose file found")

    if action == "config":
        compose_file = get_compose_file(dir_path)
        return (
            compose_file.read_text(encoding="utf-8", errors="replace").strip()
            or "No output."
        )

    # Only run status checks for actions that require it
    if action == "up" and is_compose_status(dir_path, "running"):
        raise RuntimeError("Containers are already running")
    elif action == "stop" and not is_compose_status(dir_path, "running"):
        raise RuntimeError("Containers are already stopped")
    elif action == "down" and not is_compose_status(dir_path, "stack_exists"):
        raise RuntimeError("No containers to bring down")

    outputs: list[str] = []

    # Internal helper to run docker compose commands
    def _run_cmd(args: list[str]) -> None:
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
        build_locally = is_locally_built(dir_path)
        first_step = ["build"] if build_locally else ["pull"]
        was_running = is_compose_status(dir_path, "running")
        commands_to_run = [first_step, ["down"]]
        if was_running:
            commands_to_run.append(["up", "-d", "--no-build"])

    elif action == "logs":
        commands_to_run = [["logs", "--tail", "100"]]

    elif action == "config_resolve":
        commands_to_run = [["config"]]

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
    if action not in BULK_ACTIONS:
        raise ValueError("Unsupported action")

    if action == "update":
        # Local import avoids module import cycle with dcupdate.py.
        from bot.features.dcupdate import get_system_arch

        # Compute arch once so has_dir_updates() doesn't redo it per directory.
        system_arch = get_system_arch()
        sections: list[str] = []
        for name in list_docker_dirs():
            if name in DC_IGNORE_DIRS:
                continue

            try:
                if not is_locally_built(name) and not has_dir_updates(
                    name, system_arch
                ):
                    sections.append(f"[{name}] Already up to date.")
                    continue

                output = run_single_dc("update", name)
                trimmed = tail_log_lines(output, 20).strip() or "No output."
                sections.append(f"[{name}]\n{trimmed}")
            except Exception as e:
                sections.append(f"[{name}] Failed: {e}")

        return "\n\n".join(sections) or "No output."

    sections: list[str] = []
    for name in list_docker_dirs():
        dir_path = DOCKER_APPS_DIR / name
        try:
            if action == "up" and is_compose_status(dir_path, "running"):
                sections.append(f"[{name}] Already running.")
                continue
            if action == "stop" and not is_compose_status(dir_path, "running"):
                sections.append(f"[{name}] Already stopped.")
                continue

            output = run_single_dc(action, name)
            trimmed = tail_log_lines(output, 20).strip() or "No output."
            sections.append(f"[{name}]\n{trimmed}")
        except Exception as e:
            sections.append(f"[{name}] Failed: {e}")

    return "\n\n".join(sections) or "No output."


def run_docker_prune_full() -> str:
    proc = subprocess.run(
        ["docker", "system", "prune", "-a", "-f"],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return output.strip() or "No output."


def run_docker_prune() -> str:
    commands = [
        ["docker", "container", "prune", "-f"],
        ["docker", "image", "prune", "-f"],
        ["docker", "network", "prune", "-f"],
    ]
    outputs: list[str] = []

    for cmd in commands:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        outputs.append(
            f"$ {' '.join(cmd)}\n{output}" if output else f"$ {' '.join(cmd)}"
        )
        if proc.returncode != 0:
            break

    return "\n\n".join(outputs).strip() or "No output."


def tail_log_lines(output: str, lines: int) -> str:
    if lines <= 0:
        return ""
    line_items = output.splitlines()
    if len(line_items) <= lines:
        return "\n".join(line_items)
    return "\n".join(line_items[-lines:])


def has_dir_updates(dir_name: str, system_arch: str | None = None) -> bool:
    # Local import avoids module import cycle with dcupdate.py.
    from bot.features.dcupdate import has_dir_updates as dcupdate_has_dir_updates

    return dcupdate_has_dir_updates(dir_name, system_arch)


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


def cleanup_keyboard(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🧹 Full Cleanup",
                    callback_data=dc_callback_data(
                        "cleanup",
                        user_id,
                        int(time.time()),
                        "system",
                        "ALL",
                    ),
                )
            ],
            [
                InlineKeyboardButton(
                    "🧽 Prune (Keep Cache)",
                    callback_data=dc_callback_data(
                        "cleanup",
                        user_id,
                        int(time.time()),
                        "prune",
                        "ALL",
                    ),
                )
            ],
        ]
    )


def build_action_preview(action: str, target: str | None) -> str:
    preview = (
        f"⚠️ <b>Confirm Docker Action</b>\n\n"
        f"• Action: <code>{action}</code>\n"
        f"• Target: <code>{target or 'ALL'}</code>\n\n"
        "Proceed?"
    )
    if action.startswith("config"):
        preview += (
            "\n\n<pre>Note: This may leak sensitive information, such as environment variables "
            "and secret keys. It is recommended to run this only in private chats and not in groups.</pre>"
        )
    return preview


def parse_dc_callback(raw_data: str) -> tuple[str, int, int, str, str, str] | None:
    parts = raw_data.split(":")
    if len(parts) != 7 or parts[0] != "dc":
        return None

    _, cb_type, uid, ts, action, target, sig = parts
    return cb_type, int(uid), int(ts), action, target, sig


async def validate_dc_callback(
    q, parsed_callback: tuple[str, int, int, str, str, str]
) -> bool:
    cb_type, uid, ts, action, target, sig = parsed_callback
    now = int(time.time())

    if not is_authorized_callback_user(
        getattr(q.from_user, "id", None),
        uid,
        allow_any_owner=True,
    ):
        log_security_event(
            log,
            "dcaction_callback",
            "blocked",
            detail=f"uid={getattr(q.from_user, 'id', '?')} expected={uid}",
        )
        await q.answer("🚫 Unauthorized", show_alert=True)
        return False

    if now - ts > CALLBACK_TTL:
        await q.answer(
            "⏱ This action has expired. Please run the command again.",
            show_alert=True,
        )
        return False

    payload = ":".join(["dc", cb_type, str(uid), str(ts), action, target])
    expected_sig = cb_sign(payload)

    if not hmac.compare_digest(sig, expected_sig):
        log_security_event(log, "dcaction_callback", "invalid_signature")
        await q.answer(
            "🚫 Invalid signature. Action aborted.",
            show_alert=True,
        )
        return False

    await q.answer()
    return True


async def handle_job_update_callback(q, uid: int, target: str):
    await q.edit_message_text(
        f"🐋 Running update for <code>{target}</code>…",
        parse_mode="HTML",
    )
    try:
        if not is_locally_built(target) and not has_dir_updates(target):
            log_callback(
                log, q.from_user, "dcaction", "jobup", "up_to_date", detail=target
            )
            return await q.edit_message_text(
                f"✅ <code>{target}</code> is already up to date.",
                parse_mode="HTML",
            )

        output = run_single_dc("update", target)
        log_callback(log, q.from_user, "dcaction", "jobup", "executed", detail=target)

        if len(output) > 3000:
            output = output[-3000:]

        return await q.edit_message_text(
            f"📊 <b>Docker Update Logs</b>\n"
            f"App: <code>{target}</code>\n\n"
            f"<pre>{html.escape(output)}</pre>",
            parse_mode="HTML",
            reply_markup=cleanup_keyboard(uid),
        )
    except Exception as e:
        log_callback(log, q.from_user, "dcaction", "jobup", "failed", detail=str(e))
        return await q.edit_message_text(
            f"❌ Update failed for <code>{target}</code>:\n<code>{e}</code>",
            parse_mode="HTML",
        )


async def handle_job_update_all_callback(
    q, context: ContextTypes.DEFAULT_TYPE, uid: int, token: str
):
    await q.edit_message_text("🐋 Running updates for all apps…", parse_mode="HTML")
    key = f"dcjob:{token}"
    targets = context.bot_data.get(key)
    if not targets:
        return await q.edit_message_text(
            "❌ No stored app list found for this message. Please wait for the next update check.",
            parse_mode="HTML",
        )

    context.bot_data.pop(key, None)

    total_budget = 50
    per_app_lines = max(1, total_budget // len(targets))
    sections: list[str] = []

    for app_name in targets:
        try:
            if not is_locally_built(app_name) and not has_dir_updates(app_name):
                log_callback(
                    log,
                    q.from_user,
                    "dcaction",
                    "jobupall",
                    "up_to_date",
                    detail=app_name,
                )
                sections.append(
                    f"<b>{app_name}</b>\n" "<code>Already up to date.</code>"
                )
                continue

            raw_output = run_single_dc("update", app_name)
            log_callback(
                log, q.from_user, "dcaction", "jobupall", "executed", detail=app_name
            )
            short_output = (
                tail_log_lines(raw_output, per_app_lines).strip() or "No output."
            )
            sections.append(
                f"<b>{app_name}</b>\n" f"<pre>{html.escape(short_output)}</pre>"
            )
        except Exception as e:
            log_callback(
                log,
                q.from_user,
                "dcaction",
                "jobupall",
                "failed",
                detail=f"{app_name}: {e}",
            )
            sections.append(f"<b>{app_name}</b>\n" f"<code>Update failed: {e}</code>")

    header = (
        "📊 <b>Docker Update Logs</b>\n"
        f"Updated apps: <code>{len(targets)}</code>\n"
        f"Log lines per app: <code>{per_app_lines}</code>\n\n"
    )
    max_len = 3900
    summary_parts: list[str] = [header]

    for section in sections:
        # Each section is a self-contained HTML fragment; only append
        # whole sections while staying within the Telegram size limit.
        candidate = "".join(
            summary_parts
            + (
                [section]
                if summary_parts[-1].endswith("\n\n") or not summary_parts[-1]
                else ["\n\n", section]
            )
        )
        if len(candidate) > max_len:
            break
        if summary_parts[-1].endswith("\n\n") or not summary_parts[-1]:
            summary_parts.append(section)
        else:
            summary_parts.extend(["\n\n", section])

    summary = "".join(summary_parts)
    return await q.edit_message_text(
        summary,
        parse_mode="HTML",
        reply_markup=cleanup_keyboard(uid),
    )


async def handle_cleanup_callback(q, cleanup_action: str):
    if cleanup_action == "prune":
        await q.edit_message_text("🧽 Running Docker prune (preserve build cache)…")
    else:
        await q.edit_message_text("🧹 Running Docker full cleanup…")

    try:
        if cleanup_action == "prune":
            output = run_docker_prune()
            title = "🧽 <b>Docker Prune Logs (Build Cache Preserved)</b>"
            event_action = "cleanup_prune"
        else:
            output = run_docker_prune_full()
            title = "🧹 <b>Docker Cleanup Logs (Full)</b>"
            event_action = "cleanup_full"

        log_callback(log, q.from_user, "dcaction", event_action, "executed")
        if len(output) > 3000:
            output = output[-3000:]

        return await q.edit_message_text(
            f"{title}\n\n<pre>{html.escape(output)}</pre>",
            parse_mode="HTML",
        )
    except Exception as e:
        event_action = "cleanup_prune" if cleanup_action == "prune" else "cleanup_full"
        log_callback(
            log, q.from_user, "dcaction", event_action, "failed", detail=str(e)
        )
        return await q.edit_message_text(
            f"❌ Cleanup failed:\n<code>{e}</code>",
            parse_mode="HTML",
        )


async def handle_run_callback(q, uid: int, action: str, target: str):
    await q.edit_message_text("🐋 Executing…")

    try:
        if action == "prune":
            output = run_docker_prune()
            log_callback(log, q.from_user, "dcaction", action, "executed", detail="ALL")

            if len(output) > 2000:
                output = output[-2000:]

            return await q.edit_message_text(
                "📊 <b>Docker Prune Summary (Build Cache Preserved)</b>\n\n"
                f"<pre>{html.escape(output)}</pre>",
                parse_mode="HTML",
            )

        if action == "prune_full":
            output = run_docker_prune_full()
            log_callback(log, q.from_user, "dcaction", action, "executed", detail="ALL")

            if len(output) > 2000:
                output = output[-2000:]

            return await q.edit_message_text(
                "📊 <b>Docker Prune Summary (Full)</b>\n\n"
                f"<pre>{html.escape(output)}</pre>",
                parse_mode="HTML",
            )

        if action == "update" and target != "ALL":
            if not is_locally_built(target) and not has_dir_updates(target):
                log_callback(
                    log, q.from_user, "dcaction", action, "up_to_date", detail=target
                )
                return await q.edit_message_text(
                    f"✅ <code>{target}</code> is already up to date.",
                    parse_mode="HTML",
                )

        if target == "ALL":
            output = run_bulk_dc(action)
        else:
            output = run_single_dc(action, target)
        log_callback(log, q.from_user, "dcaction", action, "executed", detail=target)

        if len(output) > 2000:
            output = output[-2000:]

        reply_markup = cleanup_keyboard(uid) if action == "update" else None

        await q.edit_message_text(
            f"📊 <b>Execution Summary</b>\n\n<pre>{html.escape(output)}</pre>",
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
    except Exception as e:
        log_callback(log, q.from_user, "dcaction", action, "failed", detail=str(e))
        await q.edit_message_text(
            f"❌ Error:\n<code>{e}</code>",
            parse_mode="HTML",
        )


# ================= Handlers =================


def dcaction_help() -> str:
    return (
        "🐋 <b>Docker Compose Action Command</b>\n\n"
        "Use <code>/dcaction</code> to manage your Docker Compose applications.\n\n"
        "<b>Available Commands:</b>\n"
        "‣ <code>/dcaction list</code>\n"
        "‣ <code>/dcaction config &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction config &lt;dir&gt; --resolve</code>\n"
        "‣ <code>/dcaction pull &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction build &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction start &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction start --all</code>\n"
        "‣ <code>/dcaction stop &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction stop --all</code>\n"
        "‣ <code>/dcaction down &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction update &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction update --all</code>\n"
        "‣ <code>/dcaction prune</code>\n"
        "‣ <code>/dcaction prune --full</code>\n"
        "‣ <code>/dcaction logs &lt;dir&gt;</code>\n"
        "‣ <code>/dcaction restart &lt;dir&gt;</code>"
        "\n\n"
        "<b>Note:</b> Replace <code>&lt;dir&gt;</code> with the name of your "
        "docker app directory as listed in <code>/dcaction list</code>."
    )


@restricted
async def dcaction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    pending_message = None
    # Check if docker CLI is available
    if not shutil.which("docker"):
        return await update.message.reply_text(
            "❌ Docker CLI not found on this system."
        )

    # Check if docker apps directory exists
    if not DOCKER_APPS_DIR.exists():
        return await update.message.reply_text(
            "❌ Docker apps directory not found. Please create it first.",
            parse_mode="HTML",
        )

    # If no arguments provided, show docker action help
    if not args:
        return await update.message.reply_text(
            "❌ Missing arguments.\n" "Run <code>/dcaction help</code> for usage info.",
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
    raw_action = args[0].lower()
    action = "up" if raw_action == "start" else raw_action
    if action not in DC_ALLOWED_ACTIONS:
        supported_actions = ["start" if a == "up" else a for a in DC_ALLOWED_ACTIONS]
        return await update.message.reply_text(
            f"❌ Supported actions: {', '.join(supported_actions)}."
        )

    # validate if --all is supported for this action
    is_all = "--all" in args
    resolve_config = False

    if action == "prune":
        is_full = "--full" in args
        if is_all or (len(args) > 1 and not is_full) or (is_full and len(args) > 2):
            return await update.message.reply_text(
                "❌ Use <code>/dcaction prune</code> or <code>/dcaction prune --full</code>.",
                parse_mode="HTML",
            )

        prune_action = "prune_full" if is_full else "prune"
        user_id = update.effective_user.id
        ts = int(time.time())
        return await update.message.reply_text(
            build_action_preview(prune_action, "ALL"),
            parse_mode="HTML",
            reply_markup=dc_keyboard(user_id, ts, prune_action, "ALL"),
        )

    if is_all and action not in BULK_ACTIONS:
        allowed_with_all = ["start" if a == "up" else a for a in sorted(BULK_ACTIONS)]
        return await update.message.reply_text(
            f"❌ The <code>--all</code> option is not supported for "
            f"<code>{raw_action}</code>.\n\n"
            f"Allowed with: {', '.join(allowed_with_all)}",
            parse_mode="HTML",
        )

    if action == "config":
        config_tokens = args[1:]
        invalid_flags = [
            t for t in config_tokens if t.startswith("--") and t != "--resolve"
        ]
        if invalid_flags:
            return await update.message.reply_text(
                "❌ Use <code>/dcaction config &lt;dir&gt;</code> or "
                "<code>/dcaction config &lt;dir&gt; --resolve</code>.",
                parse_mode="HTML",
            )
        non_flag_tokens = [t for t in config_tokens if not t.startswith("--")]
        if len(non_flag_tokens) != 1:
            return await update.message.reply_text(
                "❌ Use <code>/dcaction config &lt;dir&gt;</code> or "
                "<code>/dcaction config &lt;dir&gt; --resolve</code>.",
                parse_mode="HTML",
            )
        target = non_flag_tokens[0]
        resolve_config = "--resolve" in config_tokens
    else:
        target = None if is_all else (args[1] if len(args) > 1 else None)

    # show error if both --all and directory specified
    if is_all and len(args) > 2:
        return await update.message.reply_text(
            "❌ When using <code>--all</code>, do not specify a directory.",
            parse_mode="HTML",
        )

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

        if action == "update":
            pending_message = await update.message.reply_text("Checking for updates..")
            if not is_locally_built(target) and not has_dir_updates(target):
                return await pending_message.edit_text(
                    f"✅ <code>{target}</code> is already up to date.",
                    parse_mode="HTML",
                )

    user_id = update.effective_user.id
    ts = int(time.time())
    run_action = "config_resolve" if action == "config" and resolve_config else action
    preview_action = (
        "config --resolve" if run_action == "config_resolve" else raw_action
    )

    preview_text = build_action_preview(preview_action, target)
    preview_markup = dc_keyboard(user_id, ts, run_action, target)

    if pending_message is not None:
        await pending_message.edit_text(
            preview_text,
            parse_mode="HTML",
            reply_markup=preview_markup,
        )
        return

    await update.message.reply_text(
        preview_text,
        parse_mode="HTML",
        reply_markup=preview_markup,
    )


async def dcaction_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    parsed = parse_dc_callback(q.data)
    if parsed is None:
        log_security_event(log, "dcaction_callback", "invalid_payload")
        return await q.answer("🚫 Invalid callback", show_alert=True)

    cb_type, uid, _ts, action, target, _sig = parsed
    if not await validate_dc_callback(q, parsed):
        return

    if cb_type == "cancel":
        log_callback(log, q.from_user, "dcaction", "cancel", "cancelled", detail=target)
        return await q.edit_message_text("❌ Cancelled.")

    if cb_type == "jobup":
        return await handle_job_update_callback(q, uid, target)

    if cb_type == "jobupall":
        return await handle_job_update_all_callback(q, context, uid, target)

    if cb_type == "cleanup":
        return await handle_cleanup_callback(q, action)

    if cb_type != "run":
        return await q.answer("🚫 Invalid callback type", show_alert=True)

    await handle_run_callback(q, uid, action, target)


def get_help_section() -> str:
    return (
        "‣ <code>/dcaction</code> — Manage Docker Compose apps\n"
        "    ├ <code>/dcaction list</code>\n"
        "    ├ <code>/dcaction config &lt;dir&gt;</code>\n"
        "    ├ <code>/dcaction config &lt;dir&gt; --resolve</code>\n"
        "    ├ <code>/dcaction pull &lt;dir&gt;</code>\n"
        "    ├ <code>/dcaction build &lt;dir&gt;</code>\n"
        "    ├ <code>/dcaction start &lt;dir&gt;</code>\n"
        "    ├ <code>/dcaction start --all</code>\n"
        "    ├ <code>/dcaction stop &lt;dir&gt;</code>\n"
        "    ├ <code>/dcaction stop --all</code>\n"
        "    ├ <code>/dcaction down &lt;dir&gt;</code>\n"
        "    ├ <code>/dcaction update &lt;dir&gt;</code>\n"
        "    ├ <code>/dcaction update --all</code>\n"
        "    ├ <code>/dcaction prune</code>\n"
        "    ├ <code>/dcaction logs &lt;dir&gt;</code>\n"
        "    └ <code>/dcaction restart &lt;dir&gt;</code>"
    )


def get_commands() -> list[tuple[str, str]]:
    return [("dcaction", "Manage Docker Compose applications")]


def register_handlers(app):
    app.add_handler(CommandHandler("dcaction", dcaction))
    app.add_handler(CallbackQueryHandler(dcaction_callback, pattern=r"^dc:"))
