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
import psutil
import logging
from pathlib import Path
from contextlib import suppress

from bot.config import LOG_CHANNEL_ID, DC_IGNORE_DIRS
from bot.features.fetch import run_fastfetch
from bot.features.dcupdate import check_dir_updates, get_system_arch
from bot.features.dcaction import list_docker_dirs
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# --- Alert watchdog data ---
last_alert = {"temp": 0.0, "ram": 0.0}


def _get_uptime() -> float:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except Exception:
        return 0.0


async def notify_boot_job(context: ContextTypes.DEFAULT_TYPE):
    server_uptime = _get_uptime()
    reason = "server reboot" if server_uptime < 30 else "manual restart"
    await context.bot.send_message(
        chat_id=LOG_CHANNEL_ID,
        text=f"✅ Bot started (reason: {reason})",
    )


async def watchdog_job(context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    now = time.time()
    temp_c = 0.0

    # Get first available temperature
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


async def dcupdate_job(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Checking for docker container updates...")
    try:
        system_arch = get_system_arch()
        dirs = list_docker_dirs()
        results = {}

        for app_dir in dirs:
            if app_dir in DC_IGNORE_DIRS:
                continue

            updates = check_dir_updates(app_dir, system_arch)
            if updates:
                results[app_dir] = updates

        if not results:
            logger.info("No container updates found.")
            return  # silent when nothing new

        # formatting
        header = "Container updates available"
        if len(results) == 1:
            header = "Container update available"

        text = f"🔔 <b>{header}</b>\n\n"

        for app, services in results.items():
            for service in services:
                image = service.replace("• <b>", "").replace("</b>", "")
                text += f"• <b>{app}</b> ({image})\n"
        text += "\n"

        # suggest update command
        if len(results) == 1:
            app_name = next(iter(results))
            text  += f"<i>Run <code>/dcaction update {app_name}</code> to update this app.</i>"
        else:
            text += "<i>Run <code>/dcaction update &lt;dir&gt;</code> to update the specified app.</i>"

        await context.bot.send_message(
            chat_id=LOG_CHANNEL_ID, text=text, parse_mode="HTML"
        )

    except Exception as e:
        logger.error(f"Error in dcupdate_job: {e}")
        await context.bot.send_message(
            chat_id=LOG_CHANNEL_ID,
            text=f"❌ dcupdate job error:\n<code>{e}</code>",
            parse_mode="HTML",
        )


async def live_fastfetch(context: ContextTypes.DEFAULT_TYPE):
    fetch_info = run_fastfetch()
    timestamp = time.strftime("%Y-%m-%d %I:%M:%S %p", time.localtime())
    message_text = f"```\n{fetch_info}\n```\n_Last updated: {timestamp}_"
    sent_msg_id = context.bot_data.get("fastfetch_msg_id")
    # Try editing existing message
    if sent_msg_id:
        try:
            await context.bot.edit_message_text(
                chat_id=LOG_CHANNEL_ID,
                message_id=sent_msg_id,
                text=message_text,
                parse_mode="Markdown",
            )
            return
        except Exception as e:
            print("Fastfetch edit failed:", e)
            # Delete old message id on failure
            with suppress(Exception):
                await context.bot.delete_message(
                    chat_id=LOG_CHANNEL_ID,
                    message_id=sent_msg_id,
                )

    # Send new message fallback
    msg = await context.bot.send_message(
        chat_id=LOG_CHANNEL_ID,
        text=message_text,
        parse_mode="Markdown",
    )
    with suppress(Exception):
        await msg.pin()
    # Store message ID for future edits
    context.bot_data["fastfetch_msg_id"] = msg.message_id
