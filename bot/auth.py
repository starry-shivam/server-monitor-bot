import asyncio
from functools import wraps

from telegram import Update, Message
from telegram.ext import ContextTypes

from bot.config import OWNER_IDS


# --- Restriction decorator (owner-only) ---
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


async def delete_later(msg: Message, delay: int = 3):
    try:
        await asyncio.sleep(delay)
        await msg.delete()
        if msg.reply_to_message:
            await msg.reply_to_message.delete()
    except Exception:
        pass
