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

import io
import sys
import traceback
from html import escape
from typing import Any, Callable

from telegram import Update
from telegram.ext import ContextTypes

from bot.auth import restricted


# ================= PyExec Utils =================


def _pyexec_run(code: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
    command = "".join(f"\n    {x}" for x in code.split("\n"))
    exec_locals = {}
    exec(f"def func(update, context):{command}", globals(), exec_locals)
    return exec_locals["func"](update, context)


def _pyexec_try_and_catch(func: Callable, *args: Any, **kwargs: Any) -> str:
    try:
        output = func(*args, **kwargs)
    except Exception as exc:
        output = "".join(traceback.format_exception(None, exc, exc.__traceback__))
    return output


# ================= Handler =================


@restricted
async def pyexec(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🐍 Running Python code…")

    try:
        code = update.message.text.split(None, 1)[1]
    except IndexError:
        return await msg.edit_text("❌ No code provided.")

    old_stdout = sys.stdout
    redirected = sys.stdout = io.StringIO()

    errors = _pyexec_try_and_catch(_pyexec_run, code, update, context)

    sys.stdout = old_stdout
    output = redirected.getvalue()

    text = "<b>OUTPUT</b>:\n"
    text += f"<code>{escape(output or 'No output.')}</code>\n"

    if errors:
        text += "<b>ERRORS</b>:\n" f"<code>{escape(errors)}</code>"

    if len(text) > 4096:
        await msg.edit_text("Results too large. Sending as file.")
        f = io.BytesIO(text.encode())
        await update.message.reply_document(f.getvalue(), filename="output.txt")
    else:
        await msg.edit_text(text, parse_mode="HTML")
