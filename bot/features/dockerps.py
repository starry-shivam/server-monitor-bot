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

import json
import subprocess
import textwrap
import datetime
import time
import hmac
import hashlib
import base64
from io import BytesIO
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import (
    Update,
    InputMediaPhoto,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes

from bot.auth import restricted
from bot.config import CALLBACK_SIG_SECRET

# Per-user cooldown for refresh
DOCKER_REFRESH_COOLDOWN = 10  # seconds

# Used for rate limiting refreshes
# message_id -> last refresh timestamp
_DOCKER_REFRESH_TS = OrderedDict()
_MAX_CACHE_SIZE = 15
# ================= Docker Utils =================


def _run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except FileNotFoundError:
        raise RuntimeError("Docker not found.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.output.strip() or str(e))


def _humanize_td(seconds: float) -> str:
    seconds = max(0, int(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)

    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if not parts:
        parts.append(f"{s}s")
    return " ".join(parts)


def _parse_started_at(started_at: str) -> datetime.datetime | None:
    if not started_at or started_at == "0001-01-01T00:00:00Z":
        return None
    try:
        if started_at.endswith("Z"):
            started_at = started_at[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(started_at)
    except Exception:
        return None


def _format_ports(ports: dict | None) -> str:
    if not ports:
        return "—"
    pairs = []
    for cport, mappings in ports.items():
        if not mappings:
            continue
        for m in mappings:
            hip = m.get("HostIp") or ""
            hpt = m.get("HostPort")
            if not hpt:
                continue
            src = f"{hip}:{hpt}" if hip else hpt
            pairs.append(f"{src} → {cport}")
    return ", ".join(pairs) if pairs else "—"


def _collect_docker_containers() -> list[dict]:
    # Get all IDs first
    raw = _run_cmd(["docker", "ps", "-a", "--format", "{{.ID}}"])
    ids = [x for x in raw.splitlines() if x.strip()]
    if not ids:
        return []

    containers = []
    now = datetime.datetime.now(datetime.timezone.utc)

    fmt = (
        "{{.Name}}|{{.Config.Image}}|{{.State.Status}}|"
        "{{.State.StartedAt}}|{{json .NetworkSettings.Ports}}"
    )

    output = _run_cmd(["docker", "inspect", "-f", fmt] + ids)

    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            name, image, status, started_at, ports_json = line.split("|", 4)
        except ValueError:
            name, image, status, started_at, ports_json = (line, "", "", "", "{}")

        name = name.lstrip("/")
        started_dt = _parse_started_at(started_at)
        if started_dt and started_dt.tzinfo is None:
            started_dt = started_dt.replace(tzinfo=datetime.timezone.utc)

        uptime = "—"
        if started_dt:
            diff = (now - started_dt).total_seconds()
            uptime = _humanize_td(diff)

        try:
            ports = json.loads(ports_json)
        except Exception:
            ports = {}

        containers.append(
            {
                "name": name,
                "image": image or "—",
                "status": status or "—",
                "uptime": uptime,
                "ports": _format_ports(ports),
            }
        )

    return containers


def _color_for_status(status: str):
    s = (status or "").lower()
    if "running" in s:
        return (0.133, 0.654, 0.278)  # green
    if "paused" in s:
        return (0.976, 0.659, 0.137)  # orange
    if "restarting" in s or "dead" in s or "exited" in s:
        return (0.871, 0.176, 0.149)  # red
    return (0.400, 0.400, 0.400)  # gray


def _wrap_text(s: str, width: int) -> str:
    if not s:
        return "—"
    return textwrap.fill(s, width=width, break_long_words=False, break_on_hyphens=False)


def _render_docker_table_image(rows: list[dict]) -> bytes:
    rows = sorted(rows, key=lambda x: x["name"].lower())

    plt.rcParams["font.family"] = "sans-serif"

    headers = ["S. No", "Name", "Image", "Status", "Uptime", "Ports"]
    wrap = {"Name": 28, "Image": 34, "Status": 12, "Uptime": 10, "Ports": 72}

    formatted = []
    for i, r in enumerate(rows, start=1):
        formatted.append(
            {
                "S. No": str(i),
                "Name": _wrap_text(r["name"], wrap["Name"]),
                "Image": _wrap_text(r["image"], wrap["Image"]),
                "Status": _wrap_text(r["status"], wrap["Status"]),
                "Uptime": _wrap_text(r["uptime"], wrap["Uptime"]),
                "Ports": _wrap_text(r["ports"], wrap["Ports"]),
            }
        )

    def linecount(s: str) -> int:
        return max(1, s.count("\n") + 1)

    row_line_counts = [1] + [
        max(linecount(r[k]) for k in ["Name", "Image", "Status", "Uptime", "Ports"])
        for r in formatted
    ]

    base_row_h = 0.45
    total_lines = sum(row_line_counts)
    fig_h = max(3.0, total_lines * base_row_h) + 0.5
    fig_w = 14.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    table_data = [headers] + [
        [r["S. No"], r["Name"], r["Image"], r["Status"], r["Uptime"], r["Ports"]]
        for r in formatted
    ]

    col_widths = [0.05, 0.16, 0.22, 0.10, 0.09, 0.38]

    tbl = ax.table(
        cellText=table_data,
        cellLoc="left",
        loc="upper left",
        colWidths=col_widths,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)

    header_bg = (0.14, 0.16, 0.21)
    header_fg = (1, 1, 1)
    stripe_a = (0.97, 0.97, 0.98)
    stripe_b = (1.0, 1.0, 1.0)
    edge = (0.85, 0.85, 0.88)
    black = (0, 0, 0, 1)

    ncols = len(headers)
    normalized_heights = [lc / total_lines for lc in row_line_counts]

    for r in range(len(table_data)):
        current_row_height = normalized_heights[r]
        for c in range(ncols):
            cell = tbl[r, c]
            cell.set_height(current_row_height)
            cell.set_edgecolor(edge)
            cell.set_linewidth(0.5)
            cell.PAD = 0.12
            cell.get_text().set_va("center")

            if r == 0:
                cell.set_facecolor(header_bg)
                cell.get_text().set_color(header_fg)
                cell.get_text().set_weight("bold")
                if c == 0:
                    cell.get_text().set_ha("center")
            else:
                cell.set_facecolor(stripe_a if r % 2 else stripe_b)
                cell.get_text().set_color(black)
                if c == 0:
                    cell.get_text().set_ha("center")

    # Colorize status column
    for r in range(1, len(table_data)):
        txt = tbl[r, 3].get_text()
        txt.set_color(_color_for_status(table_data[r][3]))
        txt.set_weight("bold")

    fig.canvas.draw()
    bbox = (
        tbl.get_window_extent(fig.canvas.get_renderer())
        .transformed(fig.dpi_scale_trans.inverted())
        .expanded(1.01, 1.01)
    )

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=250, bbox_inches=bbox, pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


# ================= Callback Signing =================


def docker_sign(payload: str) -> str:
    sig = hmac.new(
        CALLBACK_SIG_SECRET.encode(),
        payload.encode(),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(sig[:9]).decode().rstrip("=")


def docker_callback_data(cb_type: str, user_id: int, msg_id: int) -> str:
    payload = f"dps:{cb_type}:{user_id}:{msg_id}"
    return f"{payload}:{docker_sign(payload)}"


# ================= Keyboard =================


def docker_keyboard(user_id: int, msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔁 Refresh",
                    callback_data=docker_callback_data(
                        "refresh",
                        user_id,
                        msg_id,
                    ),
                )
            ]
        ]
    )


# ================= Command Handler =================


@restricted
async def dockerps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_photo(
        photo="https://i.postimg.cc/Cx72g78F/IMG-20251231-160224.jpg",
        caption="🔍 Inspecting available containers...",
    )

    try:
        rows = _collect_docker_containers()
        if not rows:
            raise RuntimeError("No Docker containers found.")

        img_bytes = _render_docker_table_image(rows)

        await msg.edit_media(
            InputMediaPhoto(
                media=img_bytes,
                caption="🐳 Docker Containers",
            ),
            reply_markup=docker_keyboard(
                update.effective_user.id,
                msg.message_id,
            ),
        )

    except Exception as err:
        await msg.edit_media(
            InputMediaPhoto(
                media="https://i.postimg.cc/4d7k0rdX/IMG-20251230-195209.jpg",
                caption=f"❌ {err}",
            )
        )


# ================= Callback Handler =================


async def dockerps_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    user = q.from_user

    parts = q.data.split(":")
    if len(parts) != 5 or parts[0] != "dps":
        return await q.answer("🚫 Invalid callback", show_alert=True)

    _, cb_type, uid, msg_id, sig = parts
    uid, msg_id = int(uid), int(msg_id)

    # Owner check
    if user.id != uid:
        return await q.answer("🚫 Unauthorized", show_alert=True)

    # Signature check
    payload = ":".join(parts[:-1])
    if not hmac.compare_digest(sig, docker_sign(payload)):
        return await q.answer("🚫 Invalid signature", show_alert=True)

    # Enforce rate limit
    now = int(time.time())
    last = _DOCKER_REFRESH_TS.get(msg_id, 0)
    remaining = DOCKER_REFRESH_COOLDOWN - (now - last)

    if remaining > 0:
        return await q.answer(
            f"⏳ Wait {remaining}s",
            show_alert=False,
        )

    # Update LRU timestamp
    _DOCKER_REFRESH_TS[msg_id] = now
    _DOCKER_REFRESH_TS.move_to_end(msg_id)

    # Evict oldest entry if cache grows too large
    if len(_DOCKER_REFRESH_TS) > _MAX_CACHE_SIZE:
        _DOCKER_REFRESH_TS.popitem(last=False)

    await q.answer("🔄 Refreshing…")
    await q.edit_message_caption("🔍 Refreshing container list...")

    try:
        rows = _collect_docker_containers()
        img = _render_docker_table_image(rows)

        await q.edit_message_media(
            InputMediaPhoto(
                media=img,
                caption="🐳 Docker Containers",
            ),
            reply_markup=docker_keyboard(uid, msg_id),
        )
    except Exception as e:
        await q.edit_message_caption(f"❌ {e}")
