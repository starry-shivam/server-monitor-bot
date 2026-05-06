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
import uuid
import logging
import datetime
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.config import (
    LOG_CHANNEL_ID,
    DC_IGNORE_DIRS,
    DC_IGNORE_UPDATE_NOTIF_DIRS,
    NOTIFY_DOCKER_UPDATES,
)
from bot.features.dcupdate import check_dir_updates, get_system_arch
from bot.features.dcaction import list_docker_dirs, dc_callback_data
from bot.logger import log_job

log = logging.getLogger(__name__)


async def dcupdate_job(context: ContextTypes.DEFAULT_TYPE):
    log_job(log, "dcupdate", "started")
    try:
        system_arch = get_system_arch()
        dirs = list_docker_dirs()
        results = {}

        for app_dir in dirs:
            if app_dir in DC_IGNORE_DIRS:
                continue
            if app_dir in DC_IGNORE_UPDATE_NOTIF_DIRS:
                log_job(
                    log,
                    "dcupdate",
                    "skipped",
                    detail=f"notification_suppressed:{app_dir}",
                )
                continue

            updates = check_dir_updates(app_dir, system_arch)
            if updates:
                results[app_dir] = updates

        if not results:
            log_job(log, "dcupdate", "completed", detail="no_updates")
            return  # silent when nothing new

        # formatting
        header = "Container updates available"
        if len(results) == 1:
            header = "Container update available"

        text = f"🐳 <b>{header}</b>\n\n"

        for app, updates in results.items():
            image = updates[0].split("(")[-1].rstrip(")")
            text += f"‣ <b>{app}</b> (<code>{image}</code>)\n"
        text += "\n"

        # suggest update command
        keyboard = None
        if len(results) == 1:
            app_name = next(iter(results))
            text += f"Click the button below to update <b>{app_name}</b> or run <code>/dcaction update {app_name}</code>."
            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🔄 Update",
                            callback_data=dc_callback_data(
                                "jobup",
                                0,
                                int(time.time()),
                                "update",
                                app_name,
                            ),
                        )
                    ]
                ]
            )
        else:
            text += "Click the button below to update all apps or run <code>/dcaction update &lt;dir&gt;</code> to update the specified app."
            token = uuid.uuid4().hex[:10]
            context.bot_data[f"dcjob:{token}"] = list(results.keys())
            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🔄 Update All",
                            callback_data=dc_callback_data(
                                "jobupall",
                                0,
                                int(time.time()),
                                "update",
                                token,
                            ),
                        )
                    ]
                ]
            )

        await context.bot.send_message(
            chat_id=LOG_CHANNEL_ID,
            text=text,
            parse_mode="HTML",
            reply_markup=keyboard,
        )
        log_job(log, "dcupdate", "completed", detail=f"updates_found={len(results)}")

    except Exception as e:
        log_job(log, "dcupdate", "failed", detail=str(e))
        await context.bot.send_message(
            chat_id=LOG_CHANNEL_ID,
            text=f"❌ dcupdate job error:\n<code>{e}</code>",
            parse_mode="HTML",
        )


def register_jobs(job_queue):
    if not NOTIFY_DOCKER_UPDATES:
        return

    job_queue.run_daily(
        dcupdate_job,
        time=datetime.time(hour=1, minute=0),
        job_kwargs={"misfire_grace_time": 60},
    )
