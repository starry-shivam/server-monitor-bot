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
from telegram.ext import ContextTypes

from bot.config import LOG_CHANNEL_ID

# --- Alert watchdog data ---
last_alert = {"temp": 0.0, "ram": 0.0}


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

    # CPU temp alert (70°C) - Cooldown 30 mins
    if temp_c > 70 and (now - last_alert["temp"] > 1800):
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


def register_jobs(job_queue):
    job_queue.run_repeating(
        watchdog_job, interval=60, first=30, job_kwargs={"misfire_grace_time": 5}
    )
