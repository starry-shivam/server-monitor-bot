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

import asyncio
import hmac
import shutil
import time
import logging
from concurrent.futures import ThreadPoolExecutor

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler

from bot.auth import restricted, is_authorized_callback_user
from bot.config import CALLBACK_TTL, DOCKER_APPS_DIR, DCPANEL_START_MODE
from bot.features import cb_sign
from bot.features.dcaction import (
    list_docker_dirs,
    is_compose_status,
    run_single_dc,
)
from bot.logger import log_callback, log_security_event

log = logging.getLogger(__name__)

_GLOBAL_ACTION_RUNNING = False


def _to_base36(value: int) -> str:
    if value < 0:
        raise ValueError("base36 only supports non-negative integers")
    if value == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out: list[str] = []
    while value:
        value, remainder = divmod(value, 36)
        out.append(digits[remainder])
    return "".join(reversed(out))


def _from_base36(value: str) -> int:
    return int(value, 36)


def _app_token(name: str) -> str:
    # Deterministic short token keeps callback_data under Telegram limits.
    return cb_sign(f"app:{name}")


def _find_app_by_token(apps: list[dict[str, str]], token: str) -> dict[str, str] | None:
    for app in apps:
        if _app_token(app["name"]) == token:
            return app
    return None


def _find_app_name_by_token(token: str) -> str | None:
    for name in list_docker_dirs():
        if _app_token(name) == token:
            return name
    return None


def _resolve_start_action(action: str) -> str:
    if action == "up" and DCPANEL_START_MODE == "restart":
        return "restart"
    return action


def _status_for_dir(name: str) -> str:
    dir_path = DOCKER_APPS_DIR / name
    if is_compose_status(dir_path, "running"):
        return "running"
    if is_compose_status(dir_path, "stack_exists"):
        return "stopped"
    return "removed"


def _status_emoji(status: str) -> str:
    if status == "running":
        return "🟢"
    if status == "stopped":
        return "🔴"
    return "⚪"


def _next_action_for_status(status: str) -> tuple[str, str, str]:
    if status == "running":
        return "stop", "stop", "Stop"
    return "up", "start", "Start"


def _collect_apps() -> list[dict[str, str]]:
    names = list_docker_dirs()
    if not names:
        return []

    workers = min(8, len(names))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        statuses = list(executor.map(_status_for_dir, names))

    return [{"name": name, "status": status} for name, status in zip(names, statuses)]


def _panel_text(apps: list[dict[str, str]]) -> str:
    running_count = sum(1 for x in apps if x["status"] == "running")
    stopped_count = sum(1 for x in apps if x["status"] == "stopped")
    removed_count = sum(1 for x in apps if x["status"] == "removed")
    total = len(apps)
    return (
        "🐋 <b>Docker Control Panel</b>\n\n"
        "Tap an app to choose action, then confirm.\n"
        "🟢 Running  🔴 Stopped  ⚪ Removed\n\n"
        f"Apps: <code>{total}</code> | "
        f"Running: <code>{running_count}</code> | "
        f"Stopped: <code>{stopped_count}</code> | "
        f"Removed: <code>{removed_count}</code>"
    )


def _cb_data(kind: str, uid: int, ts: int, arg: str = "-") -> str:
    payload = ":".join(
        ["dcp", kind, _to_base36(uid), _to_base36(ts), arg]
    )
    return f"{payload}:{cb_sign(payload)}"


def _parse_callback(data: str) -> tuple[str, int, int, str, str] | None:
    try:
        payload, sig = data.rsplit(":", 1)
    except ValueError:
        return None

    parts = payload.split(":", 4)
    if len(parts) != 5 or parts[0] != "dcp":
        return None

    _, kind, uid, ts, arg = parts
    try:
        return (
            kind,
            _from_base36(uid),
            _from_base36(ts),
            arg,
            sig,
        )
    except ValueError:
        return None


def _validate_callback_signature(parsed: tuple[str, int, int, str, str]) -> bool:
    kind, uid, ts, arg, sig = parsed
    payload = ":".join(
        ["dcp", kind, _to_base36(uid), _to_base36(ts), arg]
    )
    return hmac.compare_digest(sig, cb_sign(payload))


def _panel_keyboard(uid: int, ts: int, apps: list[dict[str, str]]):
    rows: list[list[InlineKeyboardButton]] = []
    current_row: list[InlineKeyboardButton] = []

    for app in apps:
        status = app["status"]
        current_row.append(
            InlineKeyboardButton(
                f"{_status_emoji(status)} {app['name']}",
                callback_data=_cb_data(
                    "pick", uid, ts, f"app|{_app_token(app['name'])}|{status}"
                ),
            )
        )
        if len(current_row) == 2:
            rows.append(current_row)
            current_row = []

    if current_row:
        rows.append(current_row)

    rows.append(
        [
            InlineKeyboardButton(
                "▶ Start All",
                callback_data=_cb_data("pick", uid, ts, "all_up"),
            ),
            InlineKeyboardButton(
                "⏹ Stop All",
                callback_data=_cb_data("pick", uid, ts, "all_stop"),
            ),
        ]
    )
    rows.append(
        [
            InlineKeyboardButton(
                "🔄 Refresh Status",
                callback_data=_cb_data("refresh", uid, ts, "-"),
            )
        ]
    )
    return InlineKeyboardMarkup(rows)


def _confirm_keyboard(uid: int, ts: int, action_arg: str):
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Confirm",
                    callback_data=_cb_data("run", uid, ts, action_arg),
                ),
                InlineKeyboardButton(
                    "❌ Cancel",
                    callback_data=_cb_data("back", uid, ts, "-"),
                ),
            ]
        ]
    )


async def _render_panel(
    q,
    uid: int,
    *,
    notice: str | None = None,
) -> None:
    apps = await asyncio.to_thread(_collect_apps)

    panel_text = _panel_text(apps)
    if notice:
        panel_text += f"\n\n{notice}"

    await q.edit_message_text(
        panel_text,
        parse_mode="HTML",
        reply_markup=_panel_keyboard(uid, int(time.time()), apps),
    )


async def _run_for_all(action: str, apps: list[dict[str, str]]) -> tuple[int, int]:
    ran = 0
    skipped = 0
    run_action = _resolve_start_action(action)
    for app in apps:
        status = app["status"]
        name = app["name"]

        if action == "up" and status == "running":
            skipped += 1
            continue
        if action == "stop" and status != "running":
            skipped += 1
            continue

        try:
            await asyncio.to_thread(run_single_dc, run_action, name)
            ran += 1
        except Exception:
            skipped += 1
    return ran, skipped


@restricted
async def dcpanel(update: Update, _context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if _GLOBAL_ACTION_RUNNING:
        return await update.message.reply_text(
            "⏳ A Docker panel action is already running. Please wait for it to finish."
        )

    if not shutil.which("docker"):
        return await update.message.reply_text(
            "❌ Docker CLI not found on this system."
        )

    if not DOCKER_APPS_DIR.exists():
        return await update.message.reply_text(
            "❌ Docker apps directory not found. Please create it first."
        )

    msg = await update.message.reply_text(
        "⏳ Building Docker panel...",
        parse_mode="HTML",
    )

    apps = await asyncio.to_thread(_collect_apps)
    if not apps:
        await asyncio.sleep(
            0.8
        )  # So we don't immediately delete the "Building..." message and cause a jarring flash
        return await msg.edit_text("❌ No docker app directories found.")

    # Build keyboard and render in place.
    await msg.edit_text(
        _panel_text(apps),
        parse_mode="HTML",
        reply_markup=_panel_keyboard(user_id, int(time.time()), apps),
    )


async def dcpanel_callback(update: Update, _context: ContextTypes.DEFAULT_TYPE):
    global _GLOBAL_ACTION_RUNNING

    q = update.callback_query
    parsed = _parse_callback(q.data or "")
    if parsed is None:
        log_security_event(log, "dcpanel_callback", "invalid_payload")
        return await q.answer("🚫 Invalid callback", show_alert=True)

    kind, uid, ts, arg, _sig = parsed
    now = int(time.time())

    if not is_authorized_callback_user(
        getattr(q.from_user, "id", None),
        uid,
        allow_any_owner=False,
    ):
        log_security_event(
            log,
            "dcpanel_callback",
            "blocked",
            detail=f"uid={getattr(q.from_user, 'id', '?')} expected={uid}",
        )
        return await q.answer("🚫 Unauthorized", show_alert=True)

    if now - ts > CALLBACK_TTL:
        return await q.answer(
            "⏱ Panel action expired. Run /dcpanel again.",
            show_alert=True,
        )

    if not _validate_callback_signature(parsed):
        log_security_event(log, "dcpanel_callback", "invalid_signature")
        return await q.answer("🚫 Invalid signature", show_alert=True)

    if _GLOBAL_ACTION_RUNNING:
        return await q.answer(
            "⏳ A Docker panel action is already running. Please wait.",
            show_alert=True,
        )

    if kind == "back":
        await q.answer("Cancelling, please wait...")
        log_callback(log, q.from_user, "dcpanel", "cancel", "cancelled")
        return await _render_panel(q, uid)

    if kind == "refresh":
        await q.answer("Refreshing panel status...")
        log_callback(log, q.from_user, "dcpanel", "refresh", "executed")
        return await _render_panel(q, uid, notice="✅ Status refreshed.")

    if kind == "pick":
        if arg in {"all_up", "all_stop"}:
            await q.answer("Loading confirmation...")
            action = "up" if arg == "all_up" else "stop"
            if action == "up" and DCPANEL_START_MODE == "restart":
                label = "Restart all"
            else:
                label = "Start all" if action == "up" else "Stop all"
            return await q.edit_message_text(
                "⚠️ <b>Confirm Docker Action</b>\n\n"
                f"• Target: <code>ALL</code>\n"
                f"• Action: <code>{label}</code>\n\n"
                "Proceed?",
                parse_mode="HTML",
                reply_markup=_confirm_keyboard(uid, int(time.time()), arg),
            )

        parts = arg.split("|", 2)
        if len(parts) != 3 or parts[0] != "app":
            return await q.answer("🚫 Invalid app selection", show_alert=True)

        token = parts[1]
        status = parts[2]
        name = await asyncio.to_thread(_find_app_name_by_token, token)
        if not name:
            return await q.answer("🚫 App not found. Reopen panel.", show_alert=True)

        await q.answer("Loading confirmation...")
        _, short_action, display_action = _next_action_for_status(status)
        if short_action == "start" and DCPANEL_START_MODE == "restart":
            display_action = "Restart"

        return await q.edit_message_text(
            "⚠️ <b>Confirm Docker Action</b>\n\n"
            f"• App: <code>{name}</code>\n"
            f"• Current Status: <code>{status}</code>\n"
            f"• Action: <code>{display_action}</code>\n\n"
            "Proceed?",
            parse_mode="HTML",
            reply_markup=_confirm_keyboard(
                uid,
                int(time.time()),
                f"one|{token}|{short_action}",
            ),
        )

    if kind != "run":
        return await q.answer("🚫 Invalid callback type", show_alert=True)

    if _GLOBAL_ACTION_RUNNING:
        return await q.answer(
            "⏳ A Docker panel action is already running. Please wait.",
            show_alert=True,
        )

    _GLOBAL_ACTION_RUNNING = True
    try:
        await q.answer("Executing action, please wait...")
        await q.edit_message_text(
            "⏳ Executing Docker action...\n" "Please wait, this can take a while.",
            parse_mode="HTML",
        )

        if arg in {"all_up", "all_stop"}:
            action = "up" if arg == "all_up" else "stop"
            apps = await asyncio.to_thread(_collect_apps)
            ran, skipped = await _run_for_all(action, apps)
            log_callback(
                log,
                q.from_user,
                "dcpanel",
                f"{action}_all",
                "executed",
                detail=f"ran={ran} skipped={skipped}",
            )

        else:
            parts = arg.split("|")
            if len(parts) != 3 or parts[0] != "one":
                return await q.edit_message_text("❌ Invalid action payload.")

            token = parts[1]
            short_action = parts[2]
            action = "up" if short_action == "start" else "stop"
            run_action = _resolve_start_action(action)

            target = await asyncio.to_thread(_find_app_name_by_token, token)
            if not target:
                return await q.edit_message_text(
                    "❌ App not found. Run /dcpanel again."
                )

            await asyncio.to_thread(run_single_dc, run_action, target)
            log_callback(
                log,
                q.from_user,
                "dcpanel",
                run_action,
                "executed",
                detail=target,
            )

        await _render_panel(q, uid, notice="✅ Action complete.")
    except Exception as e:
        log_callback(log, q.from_user, "dcpanel", "run", "failed", detail=str(e))
        await q.edit_message_text(
            "❌ Docker action failed.\n" f"<code>{e}</code>",
            parse_mode="HTML",
        )
    finally:
        _GLOBAL_ACTION_RUNNING = False


def get_help_section() -> str:
    return "‣ <code>/dcpanel</code> — Docker panel with start/stop inline controls"


def get_commands() -> list[tuple[str, str]]:
    return [("dcpanel", "Docker panel to start/stop apps with 2 clicks")]


def register_handlers(app):
    app.add_handler(CommandHandler("dcpanel", dcpanel))
    app.add_handler(CallbackQueryHandler(dcpanel_callback, pattern=r"^dcp:"))
