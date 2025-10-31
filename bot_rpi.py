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

# This is a modified version of bot.py for Raspberry Pi devices.
# It focuses on Raspberry Pi 5 and uses fastfetch for system information.
# Additional commands like /powerc show power consumption in watts and amperes.
# It also displays voltage levels for each power rail in real time.

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


# Runs via job queue every second
async def stats_sampler_job(context: ContextTypes.DEFAULT_TYPE):
    system_stats["cpu"] = psutil.cpu_percent(interval=None)
    system_stats["mem"] = psutil.virtual_memory().percent
    system_stats["disk"] = psutil.disk_usage("/").percent


# --- OS info utilities ---
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


# ============= Command Handlers =============


# --- /start command ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    is_owner = bool(user and user.id == OWNER_ID)
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
    app.add_handler(CommandHandler("cputemp", cputemp))
    app.add_handler(CommandHandler("ip", ip))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("shell", shell))
    app.add_handler(CommandHandler("pyexec", pyexec))
    app.add_handler(CommandHandler("powerc", powerc))
    app.job_queue.run_repeating(stats_sampler_job, interval=1.0, first=0.0)

    print("🤖 Bot is running…")
    app.run_polling()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped.")
