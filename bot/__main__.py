import io
import sys
import time
import asyncio
import requests as r
import psutil
from html import escape

from telegram import Update, Message
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    JobQueue,
)

from bot.auth import restricted
from bot.config import BOT_TOKEN, OWNER_IDS
from bot.jobs import notify_boot_job, watchdog_job

# Feature handlers (will exist next)
from bot.features.fetch import fetch
from bot.features.dockerps import dockerps
from bot.features.dcaction import dcaction, dcaction_callback
from bot.features.powerc import powerc
from bot.features.metrics import metrics
from bot.features.shell import shell
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
        "    └ <code>/dcaction restart &lt;dir&gt;</code>",
    ]

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

    app = ApplicationBuilder().token(BOT_TOKEN).job_queue(JobQueue()).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("fetch", fetch))
    app.add_handler(CommandHandler("dockerps", dockerps))
    app.add_handler(CommandHandler("dcaction", dcaction))
    app.add_handler(CommandHandler("powerc", powerc))
    app.add_handler(CommandHandler("metrics", metrics))
    app.add_handler(CommandHandler("shell", shell))
    app.add_handler(CommandHandler("pyexec", pyexec))
    app.add_handler(CallbackQueryHandler(dcaction_callback, pattern=r"^dc:"))

    app.job_queue.run_once(
        notify_boot_job, when=0.5, job_kwargs={"misfire_grace_time": None}
    )
    app.job_queue.run_repeating(
        watchdog_job, interval=60, first=30, job_kwargs={"misfire_grace_time": 5}
    )

    print("🤖 Bot is running (python -m bot)…")
    app.run_polling()


if __name__ == "__main__":
    main()
