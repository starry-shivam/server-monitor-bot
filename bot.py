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
import json
import time
import asyncio
import datetime
import traceback
import textwrap
import subprocess
import shlex
import requests as r
import psutil
import matplotlib

# Set backend to Agg for headless environments (prevents GUI errors/leaks)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io import BytesIO
from functools import wraps
from typing import Any, Callable
from pathlib import Path
from html import escape

from telegram import Update, InputFile, Message
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, JobQueue

# --- Configuration ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_IDS = {int(x) for x in os.getenv("OWNER_IDS", "0").split(",") if x.strip()}
LOG_CHANNEL_ID = int(os.getenv("LOG_CHANNEL_ID", "0"))

SHELL_DENYLIST = {
    "sudo","rm","mv","shutdown","reboot","init", "systemctl",
    "service","bash","zsh","sh", "source","pkill","killall",
}
SHELL_TIMEOUT = 10  # seconds
SHELL_MAX_OUTPUT = 3600

BOT_START_TIME = time.time()

# --- Pre-compiled Regex ---
RE_PMIC_CURRENT = re.compile(r"(\S+)_A.*?=([\d.]+)A")
RE_PMIC_VOLTAGE = re.compile(r"(\S+)_V.*?=([\d.]+)V")
RE_THROTTLE_HEX = re.compile(r"0x([0-9A-Fa-f]+)")


# --- Restriction decorator (owner-only) ---
def restricted(func: Callable):
    @wraps(func)
    async def wrapped(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args: Any, **kwargs: Any
    ):
        user = update.effective_user
        if not user or user.id not in OWNER_IDS:
            msg = await update.message.reply_text("🚫 This command is owner-only.")
            # Non-blocking delete
            context.application.create_task(delete_later(msg))
            return
        return await func(update, context, *args, **kwargs)

    return wrapped


async def delete_later(msg: Message, delay: int = 3):
    try:
        await asyncio.sleep(delay)
        await msg.delete()
        if msg.reply_to_message:
            await msg.reply_to_message.delete()
    except Exception:
        pass


# ================== Job Queues =================

# --- Alert watchdog data ---
last_alert = {"temp": 0.0, "ram": 0.0}


async def notify_boot_job(context: ContextTypes.DEFAULT_TYPE):
    server_uptime = get_uptime()
    reason = "server reboot" if server_uptime < 30 else "manual restart"
    await context.bot.send_message(
        chat_id=LOG_CHANNEL_ID, text=f"✅ Bot started (reason: {reason})"
    )


async def watchdog_job(context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    now = time.time()
    temp_c = 0.0

    # Efficiently get first available temperature
    temps = psutil.sensors_temperatures()
    for entries in temps.values():
        for e in entries:
            if e.current:
                temp_c = e.current
                break
        if temp_c:
            break

    mem_pct = psutil.virtual_memory().percent

    # CPU temp alert (65°C) - Cooldown 30 mins
    if temp_c > 65 and (now - last_alert["temp"] > 1800):
        last_alert["temp"] = now
        await bot.send_message(
            chat_id=LOG_CHANNEL_ID,
            text=f"🔥 *High CPU Temp:* `{temp_c:.1f}°C`",
            parse_mode="Markdown",
        )

    # RAM alert (80%) - Cooldown 30 mins
    if mem_pct > 80 and (now - last_alert["ram"] > 1800):
        last_alert["ram"] = now
        await bot.send_message(
            chat_id=LOG_CHANNEL_ID,
            text=f"📈 *High RAM Usage:* `{mem_pct:.1f}%`",
            parse_mode="Markdown",
        )


# ============= OS info utilities =============


def run_fastfetch(include_ip: bool = False) -> str:
    structure_parts = [
        "Title","Separator","OS","Host","Kernel","Uptime","Packages","Shell","Display","DE","WM","Theme",
        "Icons","Font","Terminal","CPU","GPU","Memory","Swap","Disk","Battery","Locale","Break",
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
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        return f"Fastfetch error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def parse_pmic():
    """Reads Raspberry Pi 5 PMIC ADC rails using vcgencmd."""
    try:
        out = subprocess.check_output(["vcgencmd", "pmic_read_adc"], text=True)
    except subprocess.CalledProcessError:
        return [], 0.0

    current_map = {}
    voltage_map = {}

    for line in out.splitlines():
        if m := RE_PMIC_CURRENT.search(line):
            current_map[m.group(1)] = float(m.group(2))
        elif m := RE_PMIC_VOLTAGE.search(line):
            voltage_map[m.group(1)] = float(m.group(2))

    results = []
    total = 0.0

    for rail, amps in current_map.items():
        if rail in voltage_map:
            volts = voltage_map[rail]
            watts = amps * volts
            results.append((rail, amps, volts, watts))
            total += watts

    return results, total


def get_uptime() -> float:
    try:
        # Faster file read
        return float(Path("/proc/uptime").read_text().split()[0])
    except Exception:
        return 0.0


def get_temp():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except Exception:
        return 0.0


def get_fan():
    try:
        base = Path("/sys/class/thermal/cooling_device0")
        cur = int((base / "cur_state").read_text())
        mx = int((base / "max_state").read_text())
        return cur, mx
    except Exception:
        return None, None


def get_throttle():
    try:
        return subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
    except Exception:
        return "Unknown"


def decode_throttle(hex_str: str) -> str:
    m = RE_THROTTLE_HEX.search(hex_str)
    if not m:
        return "Unknown"

    val = int(m.group(1), 16)
    flags = []

    # Mapping of bit to message
    conditions = {
        0: "Under-voltage NOW",
        1: "Frequency capped NOW",
        2: "Currently throttled",
        3: "Soft temperature limit NOW",
    }
    history = {
        16: "Under-voltage occurred",
        17: "Frequency cap occurred",
        18: "Throttle occurred",
        19: "Soft temp limit occurred",
    }

    for bit, msg in conditions.items():
        if val & (1 << bit):
            flags.append(f"🔴 {msg} (bit {bit})")

    for bit, msg in history.items():
        if val & (1 << bit):
            flags.append(f"🟡 {msg} (bit {bit})")

    return "\n".join(flags) if flags else "🟢 All good — no throttling"


def format_power_report():
    rails, total = parse_pmic()
    temp = get_temp()
    fan_cur, fan_max = get_fan()
    throttle = get_throttle()
    decoded = decode_throttle(throttle)

    lines = [f"⚡ *Raspberry Pi 5 Power Report*\n"]
    lines.append(f"🌡Temperature: `{temp:.1f}°C`")

    if fan_cur is not None and fan_max:
        pct = fan_cur / fan_max * 100
        lines.append(f"🌀 Fan: `{fan_cur}/{fan_max}` (`{pct:.0f}%`)")

    lines.append(f"🚨 Throttle: `{throttle}`")
    lines.append(f"{decoded}\n")

    lines.append("*Rails (A × V = W):*")
    # Sort by Watts descending
    for rail, a, v, w in sorted(rails, key=lambda x: -x[3]):
        lines.append(f"`{rail:<10} {a:>5.3f}A × {v:>5.3f}V = {w:>5.3f}W`")

    lines.append(f"\n🔋 *Total Power*: `{total:.3f} W`")
    return "\n".join(lines)


def format_minimal_power_report():
    _, total = parse_pmic()
    temp = get_temp()
    fan_cur, fan_max = get_fan()
    pct = (fan_cur / fan_max * 100) if fan_max else 0
    return f"Power: `{total:.3f} W` | CPU Temp: `{temp:.1f}°C` | Fan: `{pct:.0f}%`"


# ============= Docker Utils =================


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
    # Optimization: Get all IDs first
    raw = _run_cmd(["docker", "ps", "-a", "--format", "{{.ID}}"])
    ids = [x for x in raw.splitlines() if x.strip()]
    if not ids:
        return []

    containers = []
    now = datetime.datetime.now(datetime.timezone.utc)

    # Optimization: Single inspect call for ALL containers (O(1) instead of O(N))
    fmt = "{{.Name}}|{{.Config.Image}}|{{.State.Status}}|{{.State.StartedAt}}|{{json .NetworkSettings.Ports}}"

    # Run inspect on all IDs at once
    try:
        output = _run_cmd(["docker", "inspect", "-f", fmt] + ids)
    except RuntimeError:
        # Fallback if command length is too long (unlikely)
        return []

    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            parts = line.split("|", 4)
            name, image, status, started_at, ports_json = parts
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
                "id": "—",  # ID not strictly needed for display, saves parsing logic
                "name": name,
                "image": image or "—",
                "status": status or "—",
                "uptime": uptime,
                "ports": _format_ports(ports),
            }
        )

    return containers


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
    if not s:
        return "—"
    return textwrap.fill(s, width=width, break_long_words=False, break_on_hyphens=False)


def _render_docker_table_image(rows: list[dict]) -> bytes:
    rows = sorted(rows, key=lambda x: x["name"].lower())

    # Matplotlib optimization: State machine management
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

    # Colorize status
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


# ============= Stats Utils =============


async def _stats_render_chart_bytes(
    cpu_pct: float, mem_pct: float, disk_pct: float
) -> bytes:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=(7.5, 3.5), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    fig.suptitle("System Resource Usage", fontsize=12)

    # Bar chart
    labels = ["CPU", "Disk"]
    values = [cpu_pct, disk_pct]
    ax_bar.bar(labels, values, color=["#4CAF50", "#FFC107"])
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("%")
    for i, v in enumerate(values):
        ax_bar.text(
            i, min(100, v + 2), f"{v:.1f}%", ha="center", va="bottom", fontsize=9
        )
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Pie chart
    used = max(0.0, min(100.0, mem_pct))
    ax_pie.pie(
        [used, max(0.0, 100.0 - used)],
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


# ============================================
#                Command Handlers
# ============================================


@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_name = escape(getattr(context.bot, "first_name", "Bot"))
    text = (
        f"Hi! I’m {bot_name} 🤖\n\n"
        "I can provide system information and perform various tasks on this server.\n\n"
        "Use /help to see all available commands."
    )
    await update.message.reply_text(
        text, parse_mode="HTML", disable_web_page_preview=True
    )


@restricted
async def help(update: Update, _: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    lines = [
        f"Hello {user.first_name}! Here are the available commands:\n",
        "‣ <code>/fetch</code> — Display system information using Fastfetch (-ip: include local IP)",
        "‣ <code>/dockerps</code> — Show a table of Docker containers and their statuses",
        "‣ <code>/powerc</code> — Display Pi 5 power usage, fan speed, and voltage (-v: verbose output)",
        "‣ <code>/stats</code> — Visually display CPU, RAM, and disk usage",
        "‣ <code>/ping</code> — Measure Telegram bot API latency",
        "‣ <code>/shell</code> — Execute shell commands",
        "‣ <code>/pyexec</code> — Execute Python code",
    ]
    await update.message.reply_text(
        "\n".join(lines), parse_mode="HTML", disable_web_page_preview=True
    )


@restricted
async def fetch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🛰 Gathering system info…")
    include_ip = bool(context.args and "-ip" in context.args)
    text = run_fastfetch(include_ip=include_ip)
    await msg.edit_text(f"```\n{text}\n```", parse_mode="Markdown")


@restricted
async def dockerps(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    msg = await update.message.reply_text("🔍 Inspecting available containers...")
    try:
        img_bytes = _render_docker_table_image(rows)
        await update.message.reply_photo(
            photo=InputFile(img_bytes, filename="docker_containers.png"),
            caption="🐳 Docker Containers",
        )
        await msg.delete()
    except Exception as e:
        # Fallback to text file
        try:
            payload = str(rows)  # Simple fallback since image gen failed
            await update.message.reply_document(
                io.BytesIO(payload.encode()),
                filename="docker_error_dump.txt",
                caption=f"❌ Image Render Failed: {e}",
            )
        except Exception:
            pass


@restricted
async def powerc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("📡 Reading PMIC ADC…")
    verbose = bool(context.args and "-v" in context.args)
    try:
        report = format_power_report() if verbose else format_minimal_power_report()
        await msg.edit_text(report, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Error: `{e}`", parse_mode="Markdown")


@restricted
async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🏓 Pinging Telegram API…")
    start_time = time.time()
    r.get("https://api.telegram.org", timeout=5)
    ping_time = round((time.time() - start_time) * 1000, 3)

    uptime_seconds = int(time.time() - psutil.boot_time())
    d, rem = divmod(uptime_seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)

    await msg.edit_text(
        f"🏓 Pong: `{ping_time}ms`\n🕒 Uptime: `{d}d {h}h {m}m`", parse_mode="Markdown"
    )


@restricted
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    img_bytes = await _stats_render_chart_bytes(cpu, mem, disk)
    await update.message.reply_photo(
        photo=InputFile(img_bytes, filename="stats.png"),
        caption=f"CPU: {cpu:.1f}% | RAM: {mem:.1f}% | Disk: {disk:.1f}%",
    )


# --- pyexec utilities ---
def _pyexec_run(code: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
    command = "".join(f"\n    {x}" for x in code.split("\n"))
    exec_locals = {}
    exec(f"def func(update, context):{command}", globals(), exec_locals)
    return exec_locals["func"](update, context)


def _pyexec_try_and_catch(func: Callable, *args: Any, **kwargs: Any) -> str:
    try:
        output = func(*args, **kwargs)
    except Exception as exc:
        output = "".join(traceback.format_exception(None, exc, exc.__traceback__))
    return output


# ------------------------


@restricted
async def pyexec(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🐍 Running Python code…")
    try:
        code = update.message.text.split(None, 1)[1]
    except IndexError:
        return await msg.edit_text("❌ No code provided.")
    old_stdout = sys.stdout
    redirected = sys.stdout = io.StringIO()
    errors = _pyexec_try_and_catch(_pyexec_run, code, update, context)
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


# --- shell utilities ---
def _shell_exec(command: str) -> str:
    command = command.strip()
    if not command:
        raise ValueError("Empty command")

    parts = shlex.split(command)
    if any(p in SHELL_DENYLIST for p in parts):
        raise PermissionError("Blocked command")

    try:
        proc = subprocess.run(
            parts, capture_output=True, text=True, timeout=SHELL_TIMEOUT
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return output[-SHELL_MAX_OUTPUT:] or "No output."
    except subprocess.TimeoutExpired:
        raise
    except Exception as e:
        return f"Execution Error: {str(e)}"


@restricted
async def shell(update: Update, _: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("💻 Running shell command…")
    try:
        cmd = update.message.text.split(None, 1)[1]
    except IndexError:
        return await msg.edit_text("❌ No command provided.")

    try:
        output = _shell_exec(cmd)
        await msg.edit_text(f"<pre>{escape(output)}</pre>", parse_mode="HTML")
    except PermissionError as e:
        await msg.edit_text(f"🚫 {escape(str(e))}", parse_mode="HTML")
    except subprocess.TimeoutExpired:
        await msg.edit_text("⏱ Command timed out.")
    except Exception as e:
        await msg.edit_text(f"❌ {escape(str(e))}", parse_mode="HTML")


# ============= Main Application =============
def main():
    if not BOT_TOKEN:
        print("Error: BOT_TOKEN not set.")
        return

    app = ApplicationBuilder().token(BOT_TOKEN).job_queue(JobQueue()).build()

    handlers = [
        ("start", start),
        ("help", help),
        ("fetch", fetch),
        ("dockerps", dockerps),
        ("ping", ping),
        ("stats", stats),
        ("shell", shell),
        ("pyexec", pyexec),
        ("powerc", powerc),
    ]

    for name, handler in handlers:
        app.add_handler(CommandHandler(name, handler))

    app.job_queue.run_once(
        notify_boot_job,
        when=0.5,
        job_kwargs={"misfire_grace_time": None}
    )
    app.job_queue.run_repeating(
        watchdog_job,
        interval=60, first=30,
        job_kwargs={"misfire_grace_time": 5}
    )

    print("🤖 Bot is running…")
    app.run_polling()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped.")
