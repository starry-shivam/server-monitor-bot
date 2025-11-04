#!/usr/bin/env python3
#
# MIT License
#
# Copyright (c) [2025 - Present] Stɑrry Shivɑm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import re
import io
import sys
import psutil
import time
import platform
import asyncio
import datetime
import socket
from pathlib import Path
import traceback
import subprocess
import shlex
import tempfile
import matplotlib.pyplot as plt
import requests as r
from io import BytesIO

from datetime import timedelta
from functools import wraps
from typing import Any, Callable
from html import escape

from telegram import Update, InputFile, InputMediaPhoto, Message
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, JobQueue

# --- Configuration ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_IDS = {int(x) for x in os.getenv("OWNER_IDS", "0").split(",") if x.strip()}
OWNER_USERNAME = os.getenv("OWNER_USERNAME", "")


# --- Restriction decorator (owner-only) ---
def restricted(func: Callable):
    @wraps(func)
    async def wrapped(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args: Any, **kwargs: Any
    ):
        async def delete_msg(msg: Message):
            try:
                await asyncio.sleep(3)
                await msg.delete()
                if msg.reply_to_message:
                    await msg.reply_to_message.delete()
            except Exception:
                pass

        user_id = update.effective_user.id if update.effective_user else None
        if user_id not in OWNER_IDS:
            msg = await update.message.reply_text("🚫 This command is owner-only.")
            asyncio.create_task(delete_msg(msg))
            return
        return await func(update, context, *args, **kwargs)

    return wrapped


# --- Shared system sampler for live stats ---
system_stats = {"cpu": 0, "mem": 0, "disk": 0}

# --- Alert watchdog data ---
last_alert = {"temp": 0, "ram": 0}

# ================== Job Queues =================


# Runs via job queue every second
async def stats_sampler_job(context: ContextTypes.DEFAULT_TYPE):
    system_stats["cpu"] = psutil.cpu_percent(interval=None)
    system_stats["mem"] = psutil.virtual_memory().percent
    system_stats["disk"] = psutil.disk_usage("/").percent


async def notify_boot_job(context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    chat_id = list(OWNER_IDS)[0]

    await bot.send_message(
        chat_id=chat_id, text="✅ Bot started (likely server reboot)."
    )


async def watchdog_job(context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    chat_id = list(OWNER_IDS)[0]
    now = time.time()

    # CPU temp (fallback-friendly)
    temp_c = 0.0
    temps = psutil.sensors_temperatures()
    for _, entries in temps.items():
        for e in entries:
            if e.current:
                temp_c = e.current
                break

    mem_pct = psutil.virtual_memory().percent

    # CPU temp alert (65°C)
    if temp_c > 65 and now - last_alert["temp"] > 7200:
        last_alert["temp"] = now
        await bot.send_message(
            chat_id=chat_id,
            text=f"🔥 *High CPU Temp:* `{temp_c:.1f}°C`",
            parse_mode="Markdown",
        )

    # RAM alert (80%)
    if mem_pct > 80 and now - last_alert["ram"] > 7200:
        last_alert["ram"] = now
        await bot.send_message(
            chat_id=chat_id,
            text=f"📈 *High RAM Usage:* `{mem_pct:.1f}%`",
            parse_mode="Markdown",
        )


async def daily_health_job(context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    chat_id = list(OWNER_IDS)[0]

    power = format_minimal_power_report()
    fastfetch = run_fastfetch()
    text = (
        "<b>⏰ Daily System Health Report</b>\n\n"
        f"<pre>{fastfetch}</pre>\n\n"
        f"<pre>{power}</pre>\n"
    )

    await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")


# ============= OS info utilities =============


def get_ip_address():
    try:
        response = r.get("https://api.ipify.org", timeout=5)
        response.raise_for_status()
        return response.text.strip()
    except Exception:
        return "Unavailable"


def run_fastfetch(include_ip: bool = False) -> str:
    base_structure = "Title:Separator:OS:Host:Kernel:Uptime:Packages:Shell:Display:DE:WM:WMTheme:Theme:Icons:Font:Cursor:Terminal:TerminalFont:CPU:GPU:Memory:Swap:Disk:Battery:PowerAdapter:Locale:Break"
    if include_ip:
        # Insert 'LocalIp' before 'Battery'
        structure_parts = base_structure.split(":")
        ip_index = structure_parts.index("Disk") + 1  # Place after Disk
        structure_parts.insert(ip_index, "LocalIp")
        final_structure = ":".join(structure_parts)
    else:
        final_structure = base_structure

    command = ["fastfetch", "--logo", "none", "-s", final_structure]

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    except FileNotFoundError:
        return "Fastfetch error: The 'fastfetch' command was not found."
    except subprocess.CalledProcessError as e:
        # Handle non-zero exit code errors from fastfetch itself
        return f"Fastfetch command failed with return code {e.returncode}.\nStderr: {e.stderr.strip()}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def parse_pmic():
    """
    Reads Raspberry Pi 5 PMIC ADC rails using `vcgencmd pmic_read_adc`.
    Returns:
        list of tuples (rail, amps, volts, watts)
        float total watts
    """
    out = subprocess.check_output(["vcgencmd", "pmic_read_adc"], text=True)

    current_map = {}
    voltage_map = {}

    for line in out.splitlines():
        # match current lines
        m = re.search(r"(\S+)_A.*?=([\d.]+)A", line)
        if m:
            rail = m.group(1)
            current_map[rail] = float(m.group(2))

        # match voltage lines
        m = re.search(r"(\S+)_V.*?=([\d.]+)V", line)
        if m:
            rail = m.group(1)
            voltage_map[rail] = float(m.group(2))

    results = []
    total = 0.0

    for rail, amps in current_map.items():
        if rail in voltage_map:
            volts = voltage_map[rail]
            watts = amps * volts
            results.append((rail, amps, volts, watts))
            total += watts

    return results, total


def get_temp():
    """
    Returns CPU temperature in Celsius.
    """
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return int(f.read()) / 1000.0


def get_fan():
    """
    Returns (current fan state, max fan state).
    """
    try:
        with open("/sys/class/thermal/cooling_device0/cur_state") as f:
            cur = int(f.read())
        with open("/sys/class/thermal/cooling_device0/max_state") as f:
            mx = int(f.read())
        return cur, mx
    except:
        return None, None


def get_throttle():
    """
    Returns throttling status flags.
    """
    out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True)
    return out.strip()


def decode_throttle(hex_str: str) -> str:
    """
    Decodes Raspberry Pi throttle flags like: `throttled=0x50005`
    Returns readable, colored output.
    """
    # extract hex number
    m = re.search(r"0x([0-9A-Fa-f]+)", hex_str)
    if not m:
        return "Unknown"

    val = int(m.group(1), 16)

    flags = []

    def add(bit, msg):
        if val & (1 << bit):
            flags.append(f"🔴 {msg} (bit {bit})")

    def add_prev(bit, msg):
        if val & (1 << bit):
            flags.append(f"🟡 {msg} (bit {bit})")

    # Current dangerous conditions
    add(0, "Under-voltage NOW")
    add(1, "Frequency capped NOW")
    add(2, "Currently throttled")
    add(3, "Soft temperature limit NOW")

    # Historical warnings
    add_prev(16, "Under-voltage occurred")
    add_prev(17, "Frequency cap occurred")
    add_prev(18, "Throttle occurred")
    add_prev(19, "Soft temp limit occurred")

    if not flags:
        return "🟢 All good — no throttling"

    return "\n".join(flags)


def format_power_report():
    """
    Formats all telemetry into a Markdown-friendly string.
    """
    rails, total = parse_pmic()
    temp = get_temp()
    fan_cur, fan_max = get_fan()
    throttle = get_throttle()
    decoded = decode_throttle(throttle)

    text = "⚡ *Raspberry Pi 5 Power Report*\n\n"
    text += f"🌡Temperature: `{temp:.1f}°C`\n"

    if fan_cur is not None:
        pct = (fan_cur / fan_max * 100) if fan_max else 0
        text += f"🌀 Fan: `{fan_cur}/{fan_max}` (`{pct:.0f}%`)\n"

    text += f"🚨 Throttle: `{throttle}`\n"
    text += f"{decoded}\n\n"
    text += "\n"

    text += "*Rails (A × V = W):*\n"
    for rail, a, v, w in sorted(rails, key=lambda x: -x[3]):
        text += f"`{rail:<10} {a:>5.3f}A × {v:>5.3f}V = {w:>5.3f}W`\n"

    text += "\n"
    text += f"🔋 *Total Power*: `{total:.3f} W`\n"

    return text


def format_minimal_power_report():
    """
    Prints a minimal power/thermal summary.
    """
    rails, total = parse_pmic()
    temp = get_temp()
    return f"Power: {total:.3f} W | CPU Temp: {temp:.1f}°C"


# ============= Docker Utils =================


def _run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except FileNotFoundError:
        raise RuntimeError("Docker not found. Is it installed and in PATH?")
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
    # Docker returns RFC3339 / ISO8601 with Z
    if not started_at or started_at == "0001-01-01T00:00:00Z":
        return None
    try:
        if started_at.endswith("Z"):
            started_at = started_at[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(started_at)
    except Exception:
        return None


def _format_ports(ports: dict | None) -> str:
    """
    ports structure looks like:
    {
      "80/tcp": [{"HostIp":"0.0.0.0","HostPort":"8080"}],
      "443/tcp": None
    }
    We include only bindings that have host ports.
    """
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
            dst = f"{cport}"
            src = f"{hip}:{hpt}" if hip else hpt
            pairs.append(f"{src} → {dst}")
    return ", ".join(pairs) if pairs else "—"


def _collect_docker_containers() -> list[dict]:
    # Get all container IDs
    raw = _run_cmd(["docker", "ps", "-a", "--format", "{{.ID}}"])
    ids = [x for x in raw.splitlines() if x.strip()]
    containers = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for cid in ids:
        # One inspect call per container (keeps logic simple & robust)
        # We ask only for the fields we actually use.
        fmt = (
            "{{.Name}}|{{.Config.Image}}|{{.State.Status}}|"
            "{{.State.StartedAt}}|{{json .NetworkSettings.Ports}}"
        )
        line = _run_cmd(["docker", "inspect", "-f", fmt, cid])
        try:
            name, image, status, started_at, ports_json = line.split("|", 4)
        except ValueError:
            # Fallback if some field has unexpected delimiter
            name, image, status, started_at, ports_json = (line, "", "", "", "{}")

        # Docker prepends '/' to names in inspect
        name = name.lstrip("/")
        started_dt = _parse_started_at(started_at)
        if started_dt and started_dt.tzinfo is None:
            # Assume UTC if tz missing
            started_dt = started_dt.replace(tzinfo=datetime.timezone.utc)

        uptime = "—"
        if started_dt:
            diff = (now - started_dt).total_seconds()
            uptime = _humanize_td(diff)

        try:
            import json

            ports = json.loads(ports_json)
        except Exception:
            ports = {}

        containers.append(
            {
                "id": cid,
                "name": name,
                "image": image or "—",
                "status": status or "—",
                "uptime": uptime,
                "ports": _format_ports(ports),
            }
        )
    return containers


def _render_docker_table_text(rows: list[dict]) -> str:
    if not rows:
        return "<b>No containers found.</b>"

    # Compute column widths for monospace alignment
    headers = ["Name", "Image", "Status", "Uptime", "Ports"]
    cols = {h: len(h) for h in headers}

    for r in rows:
        cols["Name"] = max(cols["Name"], len(r["name"]))
        cols["Image"] = max(cols["Image"], len(r["image"]))
        cols["Status"] = max(cols["Status"], len(r["status"]))
        cols["Uptime"] = max(cols["Uptime"], len(r["uptime"]))
        # Ports can be long—don’t hard-wrap; Telegram handles wide lines with scroll

    def pad(s, w):
        return s + " " * max(0, w - len(s))

    head = (
        f"{pad('Name', cols['Name'])}  "
        f"{pad('Image', cols['Image'])}  "
        f"{pad('Status', cols['Status'])}  "
        f"{pad('Uptime', cols['Uptime'])}  "
        f"Ports"
    )
    sep = "-" * len(head)

    lines = [head, sep]
    for r in rows:
        line = (
            f"{pad(r['name'], cols['Name'])}  "
            f"{pad(r['image'], cols['Image'])}  "
            f"{pad(r['status'], cols['Status'])}  "
            f"{pad(r['uptime'], cols['Uptime'])}  "
            f"{r['ports']}"
        )
        lines.append(line)

    return "<b>🐳 Docker Containers</b>\n\n<pre>" + escape("\n".join(lines)) + "</pre>"


def _color_for_status(status: str) -> tuple[float, float, float]:
    s = (status or "").lower()
    if "running" in s:
        return (0.133, 0.654, 0.278)  # green
    if "paused" in s:
        return (0.976, 0.659, 0.137)  # orange
    if "restarting" in s or "dead" in s or "exited" in s:
        return (0.871, 0.176, 0.149)  # red
    return (0.400, 0.400, 0.400)  # gray


def _wrap_text(s: str, width: int) -> str:
    import textwrap

    if not s:
        return "—"
    # keep arrows and ports readable; avoid breaking long tokens
    return textwrap.fill(s, width=width, break_long_words=False, break_on_hyphens=False)


def _render_docker_table_image(rows: list[dict]) -> bytes:
    """
    Render a clean, compact, colorized table using matplotlib.
    - auto height based on rows + wrapping
    - alternating row stripes
    - status chip color
    """
    import math
    from io import BytesIO
    import matplotlib.pyplot as plt

    # 1) Prepare data and wrap long fields
    headers = ["Name", "Image", "Status", "Uptime", "Ports"]

    # Heuristic wrap widths (characters) per column.
    # Ports tends to be huge; others stay modest.
    wrap = {"Name": 28, "Image": 32, "Status": 12, "Uptime": 10, "Ports": 60}

    # Copy & wrap
    formatted = []
    for r in rows:
        formatted.append(
            {
                "Name": _wrap_text(r["name"], wrap["Name"]),
                "Image": _wrap_text(r["image"], wrap["Image"]),
                "Status": _wrap_text(r["status"], wrap["Status"]),
                "Uptime": _wrap_text(r["uptime"], wrap["Uptime"]),
                "Ports": _wrap_text(r["ports"], wrap["Ports"]),
            }
        )

    # 2) Estimate required height (each wrapped line ~ one row height)
    def linecount(text: str) -> int:
        return max(1, text.count("\n") + 1)

    base_row_h = 0.48  # inches per line
    total_lines = 1  # header
    for r in formatted:
        total_lines += max(
            linecount(r["Name"]),
            linecount(r["Image"]),
            linecount(r["Status"]),
            linecount(r["Uptime"]),
            linecount(r["Ports"]),
        )

    # 3) Choose width and height
    # Width chosen to comfortably fit 5 columns w/ a big Ports column.
    fig_w = 12.5  # inches
    fig_h = min(22, max(2.5, total_lines * base_row_h))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # 4) Build 2D cell text (header + rows)
    data = [headers]
    for r in formatted:
        data.append([r["Name"], r["Image"], r["Status"], r["Uptime"], r["Ports"]])

    # Column width fractions (Ports gets most)
    # These are relative; tweak to taste.
    col_widths = [0.18, 0.22, 0.12, 0.10, 0.38]

    tbl = ax.table(
        cellText=data, loc="upper left", cellLoc="left", colWidths=col_widths
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)  # a little vertical breathing room

    # 5) Style header
    header_bg = (0.15, 0.17, 0.22)
    header_fg = (1, 1, 1)
    edge = (0.75, 0.75, 0.75)
    ncols = len(headers)
    for c in range(ncols):
        cell = tbl[0, c]
        cell.set_text_props(weight="bold", color=header_fg)
        cell.set_facecolor(header_bg)
        cell.set_edgecolor(edge)
        cell.set_linewidth(1.0)

    # 6) Style body: alternating stripes + colored status text
    stripe_a = (0.97, 0.97, 0.98)
    stripe_b = (1.00, 1.00, 1.00)

    for r in range(1, len(data)):  # skip header
        bg = stripe_a if (r % 2) else stripe_b
        for c in range(ncols):
            cell = tbl[r, c]
            cell.set_facecolor(bg)
            cell.set_edgecolor(edge)
            cell.set_linewidth(0.8)

        # status color
        status_txt = data[r][2]
        tbl[r, 2].get_text().set_color(_color_for_status(status_txt))

    # 7) Tighter layout and save
    buf = BytesIO()
    plt.tight_layout(pad=0.6)
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ============= Command Handlers =============


# --- /start command ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    is_owner = bool(user and user.id in OWNER_IDS)
    bot_name = escape(getattr(context.bot, "first_name", "Bot"))

    lines = [
        f"Hi! I’m {bot_name} 🤖",
        "I can provide system information and perform various tasks on this server.",
        "Use /help to see all available commands.",
    ]

    if not is_owner:
        lines.append(f"I’m owned and operated by @{OWNER_USERNAME}.")
        lines.append(
            'You can self-host me too — see the source on <a href="https://github.com/starry-shivam/server-monitor-bot">GitHub</a>.'
        )

    text = "\n\n".join(lines)
    await update.message.reply_text(
        text, parse_mode="HTML", disable_web_page_preview=True
    )


# --- /help command showing available commands ---
@restricted
async def help(update: Update, _: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    lines = [
        f"Hello {user.first_name}! Here are the available commands:\n",
        "<code>/info</code> — System info (Neofetch-like)",
        "<code>/info -ip</code> — Include IP address",
        "<code>/dockerps</code> — sends a formatted text table (if it fits).",
        "<code>/dockerps -img</code> — forces an image table.",
        "<code>/cputemp</code> — CPU temperature (Pi only)",
        "<code>/powerc</code> — Pi5 power usage / fan / voltage",
        "<code>/stats</code> — Show CPU/RAM/Disk usage",
        "<code>/stats -live</code> — Live system monitor",
        "<code>/ping</code> — Measure API latency",
        "<code>/ip</code> — Get server public IP",
        "<code>/shell</code> — Run shell commands",
        "<code>/pyexec</code> — Run Python code",
        "<code>/updatebot</code> — Pull from Git & restart",
    ]
    text = "\n".join(lines)
    await update.message.reply_text(
        text, parse_mode="HTML", disable_web_page_preview=True
    )


# --- /info command ---
@restricted
async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🛰 Gathering system info…")
    include_ip = len(context.args) > 0 and "-ip" in context.args
    text = run_fastfetch(include_ip=include_ip)
    await msg.edit_text(f"```\n{text}\n```", parse_mode="Markdown")


# --- /dockerps command ---
@restricted
async def dockerps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    force_img = any(arg in ("-img", "--img", "--image") for arg in context.args or [])

    try:
        rows = _collect_docker_containers()
    except RuntimeError as e:
        await update.message.reply_text(f"❌ {escape(str(e))}", parse_mode="HTML")
        return
    except Exception as e:
        await update.message.reply_text(
            f"❌ Unexpected error: {escape(str(e))}", parse_mode="HTML"
        )
        return

    if not rows:
        await update.message.reply_text("🐳 No containers found.", parse_mode="HTML")
        return

    text = _render_docker_table_text(rows)

    # Telegram hard limit ~4096 chars. Keep headroom for safety.
    if (not force_img) and len(text) <= 3800:
        await update.message.reply_text(
            text, parse_mode="HTML", disable_web_page_preview=True
        )
        return

    # Fallback to image
    try:
        img_bytes = _render_docker_table_image(rows)
        await update.message.reply_photo(
            photo=InputFile(img_bytes, filename="docker_containers.png"),
            caption="🐳 Docker Containers",
        )
    except Exception as e:
        # If image fails, at least try sending as a document text file.
        try:
            payload = text if len(text) <= 200000 else "Output too large."
            await update.message.reply_document(
                io.BytesIO(payload.encode()),
                filename="docker_containers.txt",
                caption="🐳 Docker Containers (text)",
            )
        except Exception:
            await update.message.reply_text(
                f"❌ Failed to render/send image: {escape(str(e))}", parse_mode="HTML"
            )


# --- /cputemp command ---
@restricted
async def cputemp(update: Update, _: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🌡 Reading CPU temperature…")
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            await msg.edit_text("❌ No temperature sensors found.")
            return

        for name, entries in temps.items():
            for entry in entries:
                if entry.current:
                    await msg.edit_text(f"🌡 CPU Temperature: {entry.current:.1f}°C")
                    return
        await msg.edit_text("❌ Could not read temperature values.")
    except Exception as e:
        await msg.edit_text(f"❌ Error: {e}")


@restricted
async def powerc(update: Update, _: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("📡 Reading PMIC ADC…")
    try:
        report = format_power_report()
        await msg.edit_text(report, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Error: `{e}`", parse_mode="Markdown")


# --- /ip command ---
@restricted
async def ip(update: Update, _: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🌍 Fetching public IP…")
    try:
        ip = get_ip_address()
        await msg.edit_text(f"🌐 IP Address: `{ip}`", parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Error: {e}")


# --- /ping command ---
@restricted
async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🏓 Pinging Telegram API…")
    old_time = time.time()
    r.get("https://api.telegram.org", timeout=5)
    ping_time = round((time.time() - old_time) * 1000, 3)

    uptime_seconds = int(time.time() - psutil.boot_time())
    days = uptime_seconds // 86400
    hours = (uptime_seconds % 86400) // 3600
    minutes = (uptime_seconds % 3600) // 60
    uptime_fmt = f"{days}d {hours}h {minutes}m"

    await msg.edit_text(
        f"🏓 Pong: `{ping_time}ms`\n🕒 Uptime: `{uptime_fmt}`", parse_mode="Markdown"
    )


# --- /stats command with -live support ---
@restricted
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    live = len(context.args) > 0 and "-live" in context.args

    async def render_chart_bytes(
        cpu_pct: float, mem_pct: float, disk_pct: float
    ) -> bytes:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, (ax_bar, ax_pie) = plt.subplots(
            1, 2, figsize=(7.5, 3.5), gridspec_kw={"width_ratios": [1.1, 1.0]}
        )
        fig.suptitle("System Resource Usage", fontsize=12)

        # bar for CPU and disk usage
        labels = ["CPU", "Disk"]
        values = [cpu_pct, disk_pct]
        colors = ["#4CAF50", "#FFC107"]
        ax_bar.bar(labels, values, color=colors)
        ax_bar.set_ylim(0, 100)
        ax_bar.set_ylabel("%")
        for i, v in enumerate(values):
            ax_bar.text(
                i, min(100, v + 2), f"{v:.1f}%", ha="center", va="bottom", fontsize=9
            )
        ax_bar.grid(True, axis="y", linestyle="--", alpha=0.5)

        # pie chart for ram (used vs free)
        used = max(0.0, min(100.0, mem_pct))
        free = max(0.0, 100.0 - used)
        ax_pie.pie(
            [used, free],
            labels=["Used", "Free"],
            colors=["#2196F3", "#B0BEC5"],
            autopct=lambda p: f"{p:.1f}%",
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            pctdistance=0.75,
        )
        ax_pie.set_title("Memory")

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        return buf.getvalue()

    def make_caption(i: int, n: int, cpu: float, mem: float, disk: float) -> str:
        return f"📡 Live {i}/{n}\nCPU: {cpu:.1f}% | RAM: {mem:.1f}% | Disk: {disk:.1f}%"

    if not live:
        cpu = system_stats["cpu"]
        mem = system_stats["mem"]
        disk = system_stats["disk"]
        img_bytes = await render_chart_bytes(cpu, mem, disk)
        await update.message.reply_photo(
            photo=InputFile(img_bytes, filename="stats.png"),
            caption=f"CPU: {cpu:.1f}% | RAM: {mem:.1f}% | Disk: {disk:.1f}%",
        )
        return

    # Live system stats, run for 10 updates
    total = 10
    cpu = system_stats["cpu"]
    mem = system_stats["mem"]
    disk = system_stats["disk"]
    img_bytes = await render_chart_bytes(cpu, mem, disk)
    photo_msg = await update.effective_message.reply_photo(
        photo=InputFile(img_bytes, filename="stats.png"),
        caption=make_caption(1, total, cpu, mem, disk),
    )

    for i in range(2, total + 1):
        await asyncio.sleep(1)
        cpu = system_stats["cpu"]
        mem = system_stats["mem"]
        disk = system_stats["disk"]
        img_bytes = await render_chart_bytes(cpu, mem, disk)
        media = InputMediaPhoto(
            media=img_bytes,
            filename="stats.png",
            caption=make_caption(i, total, cpu, mem, disk),
        )
        try:
            await photo_msg.edit_media(media=media)
        except BadRequest as e:
            if "message to edit not found" in str(e).lower():
                return
            raise e

    # mark as finished
    try:
        await photo_msg.edit_caption("✅ Live monitoring finished.")
    except BadRequest:
        pass


# --- Shell and Python exec utilities ---
def run(code: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
    command = "".join(f"\n    {x}" for x in code.split("\n"))
    exec_locals = {}
    exec(f"def func(update, context):{command}", globals(), exec_locals)
    return exec_locals["func"](update, context)


def try_and_catch(func: Callable, *args: Any, **kwargs: Any) -> str:
    try:
        output = func(*args, **kwargs)
    except Exception as exc:
        output = "".join(traceback.format_exception(None, exc, exc.__traceback__))
    return output


# --- /pyexec command ---
@restricted
async def pyexec(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🐍 Running Python code…")
    try:
        code = update.message.text.split(None, 1)[1]
    except IndexError:
        return await msg.edit_text("❌ No code provided.")
    old_stdout = sys.stdout
    redirected = sys.stdout = io.StringIO()
    errors = try_and_catch(run, code, update, context)
    sys.stdout = old_stdout
    output = redirected.getvalue()
    text = "<b>OUTPUT</b>:\n"
    text += f"<code>{escape(output or 'No output.')}</code>\n"
    if errors:
        text += "<b>ERRORS</b>:\n<code>{}</code>".format(escape(errors))
    if len(text) > 4096:
        await msg.edit_text("Results too large. Sending as file.")
        f = io.BytesIO(text.encode())
        await update.message.reply_document(f.getvalue(), filename="output.txt")
    else:
        await msg.edit_text(text, parse_mode="HTML")


# --- /shell command ---
@restricted
async def shell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("❌ No command provided.")
    msg = await update.message.reply_text("💻 Running shell command…")
    proc = subprocess.Popen(
        context.args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = proc.communicate()
    result = (stdout + stderr).decode().strip() or "None"
    if len(result) > 2500:
        await msg.edit_text("Results too large. Sending as file.")
        f = io.BytesIO(result.encode())
        await update.message.reply_document(f.getvalue(), filename="output.txt")
    else:
        await msg.edit_text(f"<pre>{escape(result)}</pre>", parse_mode="HTML")


# ============= Main Application =============
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).job_queue(JobQueue()).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("dockerps", dockerps))
    app.add_handler(CommandHandler("cputemp", cputemp))
    app.add_handler(CommandHandler("ip", ip))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("shell", shell))
    app.add_handler(CommandHandler("pyexec", pyexec))
    app.add_handler(CommandHandler("powerc", powerc))

    app.job_queue.run_repeating(stats_sampler_job, interval=1.0, first=0.0)
    app.job_queue.run_once(notify_boot_job, when=1)
    app.job_queue.run_repeating(watchdog_job, interval=300, first=30)
    app.job_queue.run_daily(daily_health_job, time=datetime.time(hour=9, minute=0))

    print("🤖 Bot is running…")
    app.run_polling()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped.")
