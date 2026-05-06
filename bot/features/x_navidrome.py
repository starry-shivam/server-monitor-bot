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

"""
Non-standard feature module for a specific personal Navidrome workflow.

The `x_` prefix indicates this module is intentionally outside the project's
standard/general-purpose feature set. Keep this module self-contained and do
not make other modules depend on it.
"""

import os
import html
import shutil
import logging
import subprocess
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes, CommandHandler

from bot.auth import restricted
from bot.features.dcaction import run_single_dc, tail_log_lines

log = logging.getLogger(__name__)

NAVIDROME_PLAYLIST_UPDATE_CMD = (
    os.getenv("NAVIDROME_PLAYLIST_UPDATE_CMD", "false").lower() == "true"
)
NAVIDROME_MUSIC_DIR = Path(
    os.getenv("NAVIDROME_MUSIC_DIR", "/home/starry/ssd/myfiles/music")
).expanduser()
NAVIDROME_APP_DIR = os.getenv("NAVIDROME_APP_DIR", "navidrome")

_PLAYLIST_UPDATE_SCRIPT = (
    'for dir in */; do dir="${dir%/}"; '
    'find "$dir" -type f | sort > "${dir}.m3u8"; '
    "done"
)


def _generate_playlists(music_dir: Path) -> str:
    proc = subprocess.run(
        ["bash", "-lc", _PLAYLIST_UPDATE_SCRIPT],
        cwd=music_dir,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        raise RuntimeError(output or "Playlist update command failed")

    return output or "Playlists updated successfully."


@restricted
async def update_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not NAVIDROME_PLAYLIST_UPDATE_CMD:
        return await update.message.reply_text("❌ This command is disabled.")

    if not shutil.which("bash"):
        return await update.message.reply_text("❌ bash is not available on this system.")

    if not shutil.which("docker"):
        return await update.message.reply_text("❌ Docker CLI not found on this system.")

    if not NAVIDROME_MUSIC_DIR.exists() or not NAVIDROME_MUSIC_DIR.is_dir():
        return await update.message.reply_text(
            f"❌ Music directory not found: <code>{html.escape(str(NAVIDROME_MUSIC_DIR))}</code>",
            parse_mode="HTML",
        )

    status_msg = await update.message.reply_text(
        "🎵 Updating Navidrome playlists and restarting app..."
    )

    try:
        playlist_output = _generate_playlists(NAVIDROME_MUSIC_DIR)
        restart_output = run_single_dc("restart", NAVIDROME_APP_DIR)
    except Exception as e:
        log.error("update_playlist failed: %s", e)
        return await status_msg.edit_text(
            f"❌ <b>Update failed</b>\n<code>{html.escape(str(e))}</code>",
            parse_mode="HTML",
        )

    playlist_output = tail_log_lines(playlist_output, 15)
    restart_output = tail_log_lines(restart_output, 25)

    text = (
        "✅ <b>Playlist update complete</b>\n\n"
        f"<b>Music dir:</b> <code>{html.escape(str(NAVIDROME_MUSIC_DIR))}</code>\n"
        f"<b>Docker app:</b> <code>{html.escape(NAVIDROME_APP_DIR)}</code>\n\n"
        f"<b>Playlist command output</b>\n<pre>{html.escape(playlist_output or 'No output.')}</pre>\n"
        f"<b>Docker restart output</b>\n<pre>{html.escape(restart_output or 'No output.')}</pre>"
    )

    if len(text) > 4000:
        text = (
            "✅ <b>Playlist update complete</b>\n"
            "(Output was too long and has been truncated.)\n\n"
            f"<b>Playlist command output</b>\n<pre>{html.escape(tail_log_lines(playlist_output, 8))}</pre>\n"
            f"<b>Docker restart output</b>\n<pre>{html.escape(tail_log_lines(restart_output, 12))}</pre>"
        )

    await status_msg.edit_text(text, parse_mode="HTML")


def get_help_section() -> str | None:
    if not NAVIDROME_PLAYLIST_UPDATE_CMD:
        return None
    return "‣ <code>/update_playlist</code> — Rebuild m3u8 playlists and restart Navidrome"


def get_commands() -> list[tuple[str, str]]:
    if not NAVIDROME_PLAYLIST_UPDATE_CMD:
        return []
    return [
        (
            "update_playlist",
            "Rebuild m3u8 playlists and restart Navidrome",
        )
    ]


def register_handlers(app):
    if not NAVIDROME_PLAYLIST_UPDATE_CMD:
        return

    app.add_handler(CommandHandler("update_playlist", update_playlist))
