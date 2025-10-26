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
import tempfile
import matplotlib.pyplot as plt
import requests as r
from io import BytesIO

from datetime import timedelta
from functools import wraps
from typing import Any, Callable
from html import escape

from telegram import Update, InputFile, InputMediaPhoto
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, JobQueue

# --- Configuration ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
OWNER_USERNAME = os.getenv("OWNER_USERNAME", "")


# --- Restriction decorator (owner-only) ---
def restricted(func: Callable):
    @wraps(func)
    async def wrapped(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs
    ):
        user_id = update.effective_user.id if update.effective_user else None
        if user_id != OWNER_ID:
            msg = await update.message.reply_text("🚫 This command is owner-only.")
            asyncio.create_task(msg.delete(delay=5))
            return
        return await func(update, context, *args, **kwargs)

    return wrapped


# --- Shared system sampler for live stats ---
system_stats = {"cpu": 0, "mem": 0, "disk": 0}
stop_sampler = asyncio.Event()


# Runs via job queue every second
async def stats_sampler_job(context: ContextTypes.DEFAULT_TYPE):
    system_stats["cpu"] = psutil.cpu_percent(interval=None)
    system_stats["mem"] = psutil.virtual_memory().percent
    system_stats["disk"] = psutil.disk_usage("/").percent


# --- OS info utilities ---
def get_distro():
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return platform.system()


def get_host_model():
    paths = [
        "/sys/devices/virtual/dmi/id/product_name",
        "/proc/device-tree/model",
    ]
    for p in paths:
        if Path(p).exists():
            try:
                return Path(p).read_text().strip()
            except Exception:
                pass
    return socket.gethostname()


def get_cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_shell():
    shell = os.environ.get("SHELL")
    if shell:
        return os.path.basename(shell)
    try:
        p = psutil.Process(os.getppid())
        return p.name()
    except Exception:
        return "Unknown"


def get_package_count():
    checks = [
        (
            "dpkg",
            ["bash", "-lc", "dpkg-query -f '${binary:Package}\n' -W"],
        ),  # Debian/Ubuntu
        ("pacman", ["pacman", "-Qq"]),  # Arch/Manjaro
        ("rpm", ["rpm", "-qa", "--qf", "%{NAME}\n"]),  # RHEL/Fedora/SUSE
        ("apk", ["apk", "list", "--installed"]),  # Alpine
        ("flatpak", ["flatpak", "list", "--app"]),  # Flatpak apps
        ("snap", ["snap", "list"]),  # Snap packages
    ]
    for name, cmd in checks:
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout:
                count = sum(1 for ln in res.stdout.splitlines() if ln.strip())
                return f"{count} ({name})"
        except Exception:
            pass
    # fallback to pip packages in this environment
    try:
        import pkgutil

        count = len(list(pkgutil.iter_modules()))
        return f"{count} (pip)"
    except Exception:
        return "Unavailable"


def get_ip_address():
    try:
        response = r.get("https://api.ipify.org", timeout=5)
        response.raise_for_status()
        return response.text.strip()
    except Exception:
        return "Unavailable"


def format_uptime():
    boot = datetime.datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.datetime.now() - boot
    days = uptime.days
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m"


def get_system_info(include_ip=False):
    uname = platform.uname()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu_freq = psutil.cpu_freq()
    cpu_usage = psutil.cpu_percent(interval=0.5)
    username = psutil.Process().username().split("\\")[-1]
    hostname = socket.gethostname()

    info = [
        f"{username}@{hostname}",
        "----------------------------",
        f"OS: {get_distro()}",
        f"Host: {get_host_model()}",
        f"Kernel: {uname.release}",
        f"Uptime: {format_uptime()}",
        f"Packages: {get_package_count()}",
        f"Shell: {get_shell()}",
        f"CPU: {get_cpu_model()}",
        f"CPU Usage: {cpu_usage:.1f}%",
        f"CPU Freq: {cpu_freq.current/1000:.2f} GHz",
        f"Memory: {mem.used / (1024**3):.2f}GiB / {mem.total / (1024**3):.2f}GiB",
        f"Disk: {disk.used / (1024**3):.2f}GiB / {disk.total / (1024**3):.2f}GiB",
        f"Python: {platform.python_version()}",
        f"Architecture: {platform.machine()}",
    ]
    if include_ip:
        info.append(f"IP Address: {get_ip_address()}")

    return "\n".join(info)


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
    text = get_system_info(include_ip=include_ip)
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
    uptime = timedelta(seconds=int(time.time() - psutil.boot_time()))
    await msg.edit_text(
        f"🏓 Pong: `{ping_time}ms`\n🕒 Uptime: `{uptime}`", parse_mode="Markdown"
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
    command = "".join(f"\n   {x}" for x in code.split("\n"))
    exec(f"def func(update, context):{command}")
    return locals()["func"](update, context)


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


# --- /updatebot (git pull & restart if needed) ---
@restricted
async def updatebot(update: Update, _: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔄 Checking for updates…")
    try:
        git_pull = subprocess.run(
            ["git", "pull"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output = git_pull.stdout + git_pull.stderr
        if "Already up to date" in output or "Already up-to-date" in output:
            text = f"✅ *Bot is already up to date.*\n```\n{output.strip()}\n```"
            return await msg.edit_text(text, parse_mode="Markdown")

        text = f"⬆️ *New updates pulled! Restarting bot…*\n```\n{output.strip()}\n```"
        restart = subprocess.run(
            ["sudo", "systemctl", "restart", "server-monitor-bot"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        restart_output = restart.stdout + restart.stderr
        if restart_output.strip():
            text += f"\n\n*Systemd:*\n```\n{restart_output.strip()}\n```"
        await msg.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Update failed:\n`{e}`", parse_mode="Markdown")


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
    app.add_handler(CommandHandler("updatebot", updatebot))
    app.job_queue.run_repeating(stats_sampler_job, interval=1.0, first=0.0)

    print("🤖 Bot is running…")
    app.run_polling()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_sampler.set()
        print("\n🛑 Bot stopped.")
