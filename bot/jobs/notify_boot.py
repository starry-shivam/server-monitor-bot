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

from pathlib import Path
from telegram.ext import ContextTypes

from bot.config import LOG_CHANNEL_ID


def _get_uptime() -> float:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except Exception:
        return 0.0


async def notify_boot_job(context: ContextTypes.DEFAULT_TYPE):
    server_uptime = _get_uptime()
    reason = "server reboot" if server_uptime < 60 else "manual restart"
    await context.bot.send_message(
        chat_id=LOG_CHANNEL_ID,
        text=f"✅ Bot started (reason: {reason})",
    )


def register_jobs(job_queue):
    job_queue.run_once(
        notify_boot_job, when=0.5, job_kwargs={"misfire_grace_time": None}
    )
