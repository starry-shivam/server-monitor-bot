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
import logging
import httpx
import psutil
from html import escape
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    JobQueue,
    Defaults,
)

from bot.auth import restricted
from bot.logger import log_component_event, setup_logging
from bot.config import (
    BOT_TOKEN,
    TELEGRAM_PROXY,
)
from bot.loader import (
    register_all_handlers,
    collect_help_sections,
    register_all_jobs,
    set_bot_commands,
)

log = logging.getLogger(__name__)


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
    lines = [f"Hello {user.first_name}! Here are the available commands:\n"]

    # Core commands (always available)
    lines.extend(
        [
            "‣ <code>/ping</code> — Measure Telegram API latency",
        ]
    )

    # Dynamically collect help sections from all active feature modules
    module_help_sections = collect_help_sections()

    if module_help_sections:
        lines.append("----------------------------------")
        lines.extend(module_help_sections)

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


@restricted
async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🏓 Pinging Telegram API…")
    start_time = time.time()
    with httpx.Client(proxy=TELEGRAM_PROXY or None, timeout=5.0) as client:
        client.get("https://api.telegram.org")
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
    setup_logging()

    if not BOT_TOKEN:
        print("Error: BOT_TOKEN not set.")
        return

    async def post_init(app):
        """Called after app initialization, before polling."""
        await set_bot_commands(app)

    builder = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .job_queue(JobQueue())
        .defaults(Defaults(tzinfo=ZoneInfo("Asia/Kolkata")))
        .post_init(post_init)
    )

    if TELEGRAM_PROXY:
        builder = builder.proxy(TELEGRAM_PROXY)
        builder = builder.get_updates_proxy(TELEGRAM_PROXY)

    app = builder.build()

    # Core command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help))
    app.add_handler(CommandHandler("ping", ping))

    # Dynamically register all feature module handlers
    num_modules = register_all_handlers(app)
    log_component_event(
        log,
        "bootstrap",
        "register_feature_modules",
        "completed",
        detail=f"count={num_modules}",
    )

    # Dynamically register all job modules
    num_jobs = register_all_jobs(app.job_queue)
    log_component_event(
        log,
        "bootstrap",
        "register_job_modules",
        "completed",
        detail=f"count={num_jobs}",
    )

    log_component_event(log, "bootstrap", "run_polling", "started")
    app.run_polling()


if __name__ == "__main__":
    main()
