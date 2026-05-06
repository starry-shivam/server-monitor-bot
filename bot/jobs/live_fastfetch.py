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
from contextlib import suppress
from telegram.ext import ContextTypes

from bot.config import LOG_CHANNEL_ID, LIVE_FETCH_IN_LOG
from bot.features.fetch import run_fastfetch


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


def register_jobs(job_queue):
    if not LIVE_FETCH_IN_LOG:
        return

    job_queue.run_repeating(
        live_fastfetch,
        interval=300,
        first=10,
        job_kwargs={"misfire_grace_time": 10},
    )
