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

from io import BytesIO
import time
import hmac
import logging
from collections import OrderedDict

import psutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
)
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler

from bot.features import cb_sign
from bot.auth import restricted, is_authorized_callback_user
from bot.logger import log_callback, log_security_event
from bot.config import ADDITIONAL_DRIVE_PATHS

log = logging.getLogger(__name__)

# ================= Refresh Control =================

METRICS_REFRESH_COOLDOWN = 7  # seconds
_METRICS_REFRESH_TS = OrderedDict()
_MAX_CACHE_SIZE = 15

# ================= Utils =================


def _usage_color(pct: float) -> str:
    if pct < 60:
        return "#4CAF50"
    elif pct < 80:
        return "#FFC107"
    return "#F44336"


def _gb(bytes_amount: int) -> float:
    return bytes_amount / (1024**3)


def _collect_metrics():
    disk_paths = ["/"] + ADDITIONAL_DRIVE_PATHS.copy()
    cpu_pct = psutil.cpu_percent(interval=None)
    freq = psutil.cpu_freq()
    cpu_freq_ghz = (freq.current / 1000) if freq else 0.0
    mem = psutil.virtual_memory()

    # Collect metrics for all requested disk paths
    disks_data = []
    for path in disk_paths:
        try:
            usage = psutil.disk_usage(path)
            disks_data.append(
                {
                    "path": path,
                    "pct": usage.percent,
                    "used_gb": _gb(usage.used),
                    "total_gb": _gb(usage.total),
                }
            )
        except Exception:
            # Handle cases where a config path might not exist or be unmounted
            continue

    return dict(
        cpu_pct=cpu_pct,
        cpu_freq_ghz=cpu_freq_ghz,
        mem_pct=mem.percent,
        mem_used_gb=_gb(mem.used),
        mem_total_gb=_gb(mem.total),
        disks=disks_data,  # New structure: list of dicts
    )


# ================= Chart Renderer =================


async def _metrics_render_chart(
    cpu_pct: float,
    cpu_freq_ghz: float,
    mem_pct: float,
    mem_used_gb: float,
    mem_total_gb: float,
    disks: list[dict],  # Changed from flat args to a list of dicts
) -> bytes:
    plt.style.use("seaborn-v0_8-white")

    num_disks = len(disks)

    # --- Dynamic Layout Calculation ---
    # Base height for the top rings (CPU/Mem)
    top_section_height = 3.5
    # Height per disk bar (gives enough room for label + bar + gap)
    height_per_disk = 1.2

    bottom_section_height = max(1.5, num_disks * height_per_disk)
    total_fig_height = top_section_height + bottom_section_height

    # Calculate height ratios for GridSpec
    # This ensures the top rings stay circular and the bottom area expands
    ratios = [top_section_height, bottom_section_height]

    fig = plt.figure(figsize=(8, total_fig_height))

    # Adjust hspace to separate rings from the first disk bar
    gs = fig.add_gridspec(2, 2, height_ratios=ratios, hspace=0.3)

    ax_cpu = fig.add_subplot(gs[0, 0])
    ax_mem = fig.add_subplot(gs[0, 1])
    ax_disk = fig.add_subplot(gs[1, :])

    # Title adjustment
    fig.suptitle(
        "System Resource Usage", fontsize=16, fontweight="bold", y=0.96, color="#222"
    )

    # Common props
    donut_props = dict(width=0.20, edgecolor="white", linewidth=3)
    empty_color = "#F1F3F4"
    text_color = "#333333"
    sub_text_color = "#757575"

    # ================= CPU (Unchanged) =================
    ax_cpu.pie(
        [cpu_pct, 100 - cpu_pct],
        colors=[_usage_color(cpu_pct), empty_color],
        startangle=90,
        counterclock=False,
        wedgeprops=donut_props,
    )
    ax_cpu.text(
        0,
        0.15,
        f"{cpu_pct:.1f}%",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=text_color,
    )
    ax_cpu.text(
        0,
        -0.25,
        f"{cpu_freq_ghz:.2f} GHz",
        ha="center",
        va="center",
        fontsize=11,
        color=sub_text_color,
        fontweight="medium",
    )
    ax_cpu.text(
        0,
        -1.25,
        "CPU Load",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=text_color,
    )

    # ================= MEMORY (Unchanged) =================
    ax_mem.pie(
        [mem_pct, 100 - mem_pct],
        colors=[_usage_color(mem_pct), empty_color],
        startangle=90,
        counterclock=False,
        wedgeprops=donut_props,
    )
    ax_mem.text(
        0,
        0.15,
        f"{mem_pct:.1f}%",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=text_color,
    )
    ax_mem.text(
        0,
        -0.25,
        f"{mem_used_gb:.1f}/{mem_total_gb:.1f} GB",
        ha="center",
        va="center",
        fontsize=11,
        color=sub_text_color,
        fontweight="medium",
    )
    ax_mem.text(
        0,
        -1.25,
        "Memory",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=text_color,
    )

    # ================= DISKS (Dynamic Loop) =================
    ax_disk.set_axis_off()
    ax_disk.set_xlim(0, 100)

    # Set Y-axis to accommodate N disks.
    # We invert axis so index 0 is at the top.
    ax_disk.set_ylim(0, num_disks)
    ax_disk.invert_yaxis()

    # Bar Layout Constants
    bar_height = 0.4

    for i, disk in enumerate(disks):
        pct = disk["pct"]
        path_name = disk["path"]
        used = disk["used_gb"]
        total = disk["total_gb"]

        # Determine vertical center for this row (0, 1, 2...)
        y_center = i + 0.5

        # Offsets relative to y_center
        text_y_pos = y_center - 0.25  # Text above bar
        bar_y_pos = y_center + 0.1  # Bar below text

        # 1. Path Label (Left)
        ax_disk.text(
            0,
            text_y_pos,
            f"Drive {i}: {path_name}",
            fontsize=12,
            fontweight="bold",
            color=text_color,
            va="center",
        )

        # 2. Stats Label (Right)
        ax_disk.text(
            100,
            text_y_pos,
            f"{used:.1f} / {total:.1f} GB ({pct:.1f}%)",
            ha="right",
            fontsize=11,
            color=sub_text_color,
            va="center",
        )

        # 3. Background Track
        ax_disk.barh(
            bar_y_pos, 100, height=bar_height, color=empty_color, align="center", left=0
        )

        # 4. Usage Bar
        ax_disk.barh(
            bar_y_pos,
            pct,
            height=bar_height,
            color=_usage_color(pct),
            align="center",
            left=0,
        )

    # Final margins
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    return buf.getvalue()


# ================= Callback Data =================


def metrics_callback_data(cb_type: str, user_id: int, msg_id: int) -> str:
    payload = f"mtr:{cb_type}:{user_id}:{msg_id}"
    return f"{payload}:{cb_sign(payload)}"


def metrics_keyboard(user_id: int, msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔁 Refresh",
                    callback_data=metrics_callback_data("refresh", user_id, msg_id),
                )
            ]
        ]
    )


# ================= Command Handler =================


@restricted
async def metrics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_photo(
        photo="https://i.postimg.cc/GtQMKpFc/(1).jpg",
        caption="⏳ Collecting system metrics...",
    )

    data = _collect_metrics()
    img = await _metrics_render_chart(**data)

    await msg.edit_media(
        InputMediaPhoto(media=img, caption="📊 System Resource Usage"),
        reply_markup=metrics_keyboard(update.effective_user.id, msg.message_id),
    )


# ================= Callback Handler =================


async def metrics_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    parts = q.data.split(":")

    if len(parts) != 5 or parts[0] != "mtr":
        log_security_event(log, "metrics_callback", "invalid_payload")
        return await q.answer("🚫 Invalid callback", show_alert=True)

    _, cb, uid, msg_id, sig = parts
    uid, msg_id = int(uid), int(msg_id)

    if not is_authorized_callback_user(getattr(q.from_user, "id", None), uid):
        log_security_event(
            log,
            "metrics_callback",
            "blocked",
            detail=f"uid={getattr(q.from_user, 'id', '?')} expected={uid}",
        )
        return await q.answer("🚫 Unauthorized", show_alert=True)

    payload = ":".join(parts[:-1])
    if not hmac.compare_digest(sig, cb_sign(payload)):
        log_security_event(log, "metrics_callback", "invalid_signature")
        return await q.answer("🚫 Invalid signature", show_alert=True)

    now = int(time.time())
    last = _METRICS_REFRESH_TS.get(msg_id, 0)
    wait = METRICS_REFRESH_COOLDOWN - (now - last)

    if wait > 0:
        return await q.answer(f"⏳ Wait {wait}s")

    _METRICS_REFRESH_TS[msg_id] = now
    _METRICS_REFRESH_TS.move_to_end(msg_id)
    if len(_METRICS_REFRESH_TS) > _MAX_CACHE_SIZE:
        _METRICS_REFRESH_TS.popitem(last=False)

    await q.answer("🔄 Refreshing…")
    await q.edit_message_caption(" 🔄 Refreshing metrics...")

    data = _collect_metrics()
    img = await _metrics_render_chart(**data)
    log_callback(log, q.from_user, "metrics", cb, "executed")

    await q.edit_message_media(
        InputMediaPhoto(media=img, caption="📊 System Resource Usage"),
        reply_markup=metrics_keyboard(uid, msg_id),
    )


def get_help_section() -> str:
    return "‣ <code>/metrics</code> — Visual CPU, RAM, disk usage"


def get_commands() -> list[tuple[str, str]]:
    return [("metrics", "Visual CPU, RAM, disk usage")]


def register_handlers(app):
    app.add_handler(CommandHandler("metrics", metrics))
    app.add_handler(CallbackQueryHandler(metrics_callback, pattern=r"^mtr:"))
