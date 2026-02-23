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
from functools import wraps

from telegram import Update, Message
from telegram.ext import ContextTypes

from bot.config import OWNER_IDS


def restricted(func):
    @wraps(func)
    async def wrapped(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *args,
        **kwargs,
    ):
        user = update.effective_user
        if not user or user.id not in OWNER_IDS:
            msg = await update.message.reply_text(
                "🚫 You are not authorized to use this command."
            )
            context.application.create_task(delete_later(msg))
            return
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
