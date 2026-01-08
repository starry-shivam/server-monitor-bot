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

import subprocess

from telegram import Update
from telegram.ext import ContextTypes
from bot.auth import restricted


def run_fastfetch(include_ip: bool = False) -> str:
    structure_parts = [
        "Title",
        "Separator",
        "OS",
        "Host",
        "Kernel",
        "Uptime",
        "Packages",
        "Shell",
        "Display",
        "DE",
        "WM",
        "Theme",
        "Icons",
        "Font",
        "Terminal",
        "CPU",
        "GPU",
        "Memory",
        "Swap",
        "Disk",
        "Battery",
        "Locale",
        "Break",
    ]

    if include_ip:
        # Insert 'LocalIp' after 'Disk'
        try:
            idx = structure_parts.index("Disk") + 1
            structure_parts.insert(idx, "LocalIp")
        except ValueError:
            pass

    final_structure = ":".join(structure_parts)
    command = ["fastfetch", "--logo", "none", "-s", final_structure]

    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except FileNotFoundError as e:
        return f"Fastfetch error: {e}"
    except subprocess.CalledProcessError as e:
        return f"Fastfetch error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


@restricted
async def fetch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🛰 Gathering system info…")
    include_ip = bool(context.args and "--ip" in context.args)
    text = run_fastfetch(include_ip=include_ip)
    await msg.edit_text(f"```\n{text}\n```", parse_mode="Markdown")
