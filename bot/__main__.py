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
import requests as r
import psutil
from html import escape
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    JobQueue,
    Defaults,
)

from bot.auth import restricted
from bot.config import BOT_TOKEN, POWER_MGMT_AVAILABLE
from bot.jobs import notify_boot_job, watchdog_job

# Feature imports
from bot.features.fetch import fetch, fetch_callback
from bot.features.dockerps import dockerps, dockerps_callback
from bot.features.dcaction import dcaction, dcaction_callback
from bot.features.powerc import powerc, powerc_callback
from bot.features.powerm import reboot, poweroff, power_callback
from bot.features.metrics import metrics, metrics_callback
from bot.features.shell import shell, shell_callback
from bot.features.pyexec import pyexec


@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_name = escape(getattr(context.bot, "first_name", "Bot"))
    text = (
        f"Hi! I’m {bot_name} 🤖\n\n"
        "I can provide system information and perform various tasks on this server.\n\n"
        "Use /help to see all available commands."
    )
    await update.message.reply_text(text, parse_mode="HTML")


@restricted
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    lines = [
        f"Hello {user.first_name}! Here are the available commands:\n",
        "‣ <code>/fetch</code> — Display system information using Fastfetch",
        "‣ <code>/dockerps</code> — Show Docker containers",
        "‣ <code>/powerc</code> — Display Pi 5 power usage",
        "‣ <code>/metrics</code> — Visual CPU, RAM, disk usage",
        "‣ <code>/ping</code> — Measure Telegram API latency",
        "‣ <code>/shell</code> — Execute shell commands",
        "‣ <code>/pyexec</code> — Execute Python code",
        "----------------------------------",
        "‣ <code>/dcaction</code> — Manage Docker Compose apps",
        "    ├ <code>/dcaction list</code>",
        "    ├ <code>/dcaction pull &lt;dir&gt;</code>",
        "    ├ <code>/dcaction build &lt;dir&gt;</code>",
        "    ├ <code>/dcaction up &lt;dir&gt;</code>",
        "    ├ <code>/dcaction up --all</code>",
        "    ├ <code>/dcaction stop &lt;dir&gt;</code>",
        "    ├ <code>/dcaction stop --all</code>",
        "    ├ <code>/dcaction pause &lt;dir&gt;</code>",
        "    ├ <code>/dcaction pause --all</code>",
        "    ├ <code>/dcaction unpause &lt;dir&gt;</code>",
        "    ├ <code>/dcaction unpause --all</code>",
        "    ├ <code>/dcaction update &lt;dir&gt;</code>",
        "    ├ <code>/dcaction logs &lt;dir&gt;</code>",
        "    ├ <code>/dcaction down &lt;dir&gt;</code>",
        "    └ <code>/dcaction restart &lt;dir&gt;</code>",
    ]

    if POWER_MGMT_AVAILABLE:
        lines.extend(
            [
                "----------------------------------",
                "‣ <code>/reboot</code> — Reboot the server",
                "‣ <code>/poweroff</code> — Power off the server",
            ]
        )

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


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
        f"🏓 Pong: `{ping_time}ms`\n🕒 Uptime: `{d}d {h}h {m}m`",
        parse_mode="Markdown",
    )


def main():
    if not BOT_TOKEN:
        print("Error: BOT_TOKEN not set.")
        return

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .job_queue(JobQueue())
        .defaults(Defaults(tzinfo=ZoneInfo("Asia/Kolkata")))
        .build()
    )

    # Command handlers
    COMMAND_HANDLERS = {
        "start": start,
        "help": help,
        "ping": ping,
        "fetch": fetch,
        "dockerps": dockerps,
        "dcaction": dcaction,
        "powerc": powerc,
        "metrics": metrics,
        "shell": shell,
        "pyexec": pyexec,
    }

    if POWER_MGMT_AVAILABLE:
        COMMAND_HANDLERS.update(
            {
                "reboot": reboot,
                "poweroff": poweroff,
            }
        )

    for command, handler in COMMAND_HANDLERS.items():
        app.add_handler(CommandHandler(command, handler))

    # Callback query handlers
    CALLBACK_HANDLERS = {
        r"^ffc:": fetch_callback,
        r"^dps:": dockerps_callback,
        r"^dc:": dcaction_callback,
        r"^pwc:": powerc_callback,
        r"^mtr:": metrics_callback,
        r"^sh:": shell_callback,
    }

    if POWER_MGMT_AVAILABLE:
        CALLBACK_HANDLERS.update(
            {
                r"^pw:": power_callback,
            }
        )

    for pattern, handler in CALLBACK_HANDLERS.items():
        app.add_handler(CallbackQueryHandler(handler, pattern=pattern))

    app.job_queue.run_once(
        notify_boot_job, when=0.5, job_kwargs={"misfire_grace_time": None}
    )
    app.job_queue.run_repeating(
        watchdog_job, interval=60, first=30, job_kwargs={"misfire_grace_time": 5}
    )

    print("🤖 Bot is running…")
    app.run_polling()


if __name__ == "__main__":
    main()
