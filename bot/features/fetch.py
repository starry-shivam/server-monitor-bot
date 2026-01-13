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


import subprocess
import time
import hmac
from collections import OrderedDict

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes

from bot.auth import restricted
from bot.features import cb_sign

# ================= Refresh Control =================

FETCH_REFRESH_COOLDOWN = 5  # seconds
_FETCH_REFRESH_TS = OrderedDict()
_MAX_CACHE_SIZE = 15


# ================ Fastfetch Runner ================


def run_fastfetch(include_ip: bool = False) -> str:
    structure_parts = [
        "Title",
        "Separator",
        "OS",
        "Host",
        "Kernel",
        "Uptime",
        "Packages",
        "Shell",
        "Display",
        "DE",
        "WM",
        "Theme",
        "Icons",
        "Font",
        "Terminal",
        "CPU",
        "GPU",
        "Memory",
        "Swap",
        "Disk",
        "Battery",
        "Locale",
        "Break",
    ]

    if include_ip:
        # Insert 'LocalIp' after 'Disk'
        try:
            idx = structure_parts.index("Disk") + 1
            structure_parts.insert(idx, "LocalIp")
        except ValueError:
            pass

    final_structure = ":".join(structure_parts)
    command = ["fastfetch", "--logo", "none", "-s", final_structure]

    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except FileNotFoundError as e:
        return f"Fastfetch error: {e}"
    except subprocess.CalledProcessError as e:
        return f"Fastfetch error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# ================= Callback Data =================


def fetch_callback_data(
    cb_type: str, user_id: int, msg_id: int, include_ip: int
) -> str:
    payload = f"ffc:{cb_type}:{user_id}:{msg_id}:{include_ip}"
    return f"{payload}:{cb_sign(payload)}"


def fetch_keyboard(user_id: int, msg_id: int, include_ip: bool) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔁 Refresh",
                    callback_data=fetch_callback_data(
                        "refresh", user_id, msg_id, int(include_ip)
                    ),
                )
            ]
        ]
    )


# ================= Command Handler =================


@restricted
async def fetch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    include_ip = bool(context.args and "--ip" in context.args)

    msg = await update.message.reply_text("🛰 Gathering system info…")
    text = run_fastfetch(include_ip=include_ip)

    await msg.edit_text(
        f"```\n{text}\n```",
        parse_mode="Markdown",
        reply_markup=fetch_keyboard(
            update.effective_user.id,
            msg.message_id,
            include_ip,
        ),
    )


# ================= Callback Handler =================


async def fetch_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    parts = q.data.split(":")

    if len(parts) != 6 or parts[0] != "ffc":
        return await q.answer("🚫 Invalid callback", show_alert=True)

    _, cb, uid, msg_id, include_ip, sig = parts
    uid = int(uid)
    msg_id = int(msg_id)
    include_ip = bool(int(include_ip))

    if q.from_user.id != uid:
        return await q.answer("🚫 Unauthorized", show_alert=True)

    payload = ":".join(parts[:-1])
    if not hmac.compare_digest(sig, cb_sign(payload)):
        return await q.answer("🚫 Invalid signature", show_alert=True)

    now = int(time.time())
    last = _FETCH_REFRESH_TS.get(msg_id, 0)
    wait = FETCH_REFRESH_COOLDOWN - (now - last)

    if wait > 0:
        return await q.answer(f"⏳ Wait {wait}s")

    _FETCH_REFRESH_TS[msg_id] = now
    _FETCH_REFRESH_TS.move_to_end(msg_id)
    if len(_FETCH_REFRESH_TS) > _MAX_CACHE_SIZE:
        _FETCH_REFRESH_TS.popitem(last=False)

    await q.answer("🔄 Refreshing…")
    await q.edit_message_text("🛰 Refreshing system info…")

    text = run_fastfetch(include_ip=include_ip)

    await q.edit_message_text(
        f"```\n{text}\n```",
        parse_mode="Markdown",
        reply_markup=fetch_keyboard(uid, msg_id, include_ip),
    )
