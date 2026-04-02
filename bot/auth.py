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

import asyncio
import logging
from functools import wraps

from telegram import Update, Message
from telegram.ext import ContextTypes

from bot.logger import log_command, log_security_event
from bot.config import OWNER_IDS

log = logging.getLogger(__name__)


def is_owner_user(user_id: int | None) -> bool:
    return bool(user_id) and user_id in OWNER_IDS


def is_authorized_callback_user(
    actual_user_id: int | None,
    expected_user_id: int,
    *,
    allow_any_owner: bool = False,
) -> bool:
    if not is_owner_user(actual_user_id):
        return False
    if allow_any_owner and expected_user_id == 0:
        return True
    return actual_user_id == expected_user_id


def restricted(func):
    @wraps(func)
    async def wrapped(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *args,
        **kwargs,
    ):
        user = update.effective_user
        command_name = getattr(getattr(update, "message", None), "text", func.__name__)
        command_name = (command_name or func.__name__).split()[0]

        if not user or not is_owner_user(user.id):
            log_security_event(
                log,
                "unauthorized_command",
                "blocked",
                detail=f"command={command_name} user={getattr(user, 'id', '?')}",
            )
            msg = await update.message.reply_text(
                "🚫 You are not authorized to use this command."
            )
            context.application.create_task(delete_later(msg))
            return
        log_command(log, update, command_name, "accepted")
        return await func(update, context, *args, **kwargs)

    return wrapped


# Helper function to delete messages after a delay
async def delete_later(msg: Message, delay: int = 3):
    try:
        await asyncio.sleep(delay)
        await msg.delete()
        # if msg.reply_to_message:
        #     await msg.reply_to_message.delete()
    except Exception:
        pass
