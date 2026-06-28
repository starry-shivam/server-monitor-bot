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
import asyncio
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes, CommandHandler

from bot.auth import restricted
from bot.features.dcaction import has_compose_file, tail_log_lines
from bot.config import DOCKER_APPS_DIR
from bot.logger import log_callback

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


def _scan_navidrome_library(app_name: str) -> str:
    dir_path = DOCKER_APPS_DIR / app_name

    if not dir_path.exists():
        raise FileNotFoundError("Directory does not exist")
    if not has_compose_file(dir_path):
        raise RuntimeError("No docker compose file found")

    proc = subprocess.run(
        ["docker", "compose", "exec", "navidrome", "/app/navidrome", "scan", "-f"],
        cwd=dir_path,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        raise RuntimeError(output or "Navidrome scan command failed")
    return output or "No output."


def _collect_playlist_stats(music_dir: Path) -> tuple[int, int]:
    subdirs = sorted(p for p in music_dir.iterdir() if p.is_dir())
    playlist_count = len(subdirs)
    track_count = 0

    for subdir in subdirs:
        track_count += sum(1 for p in subdir.rglob("*") if p.is_file())

    return playlist_count, track_count


def _generate_playlists(music_dir: Path) -> tuple[int, int]:
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

    playlist_count, track_count = _collect_playlist_stats(music_dir)
    return playlist_count, track_count


@restricted
async def update_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not NAVIDROME_PLAYLIST_UPDATE_CMD:
        return await update.message.reply_text("❌ This command is disabled.")

    log_callback(log, update.effective_user, "navidrome", "update_playlist", "accepted")

    if not shutil.which("bash"):
        return await update.message.reply_text(
            "❌ bash is not available on this system."
        )

    if not shutil.which("docker"):
        return await update.message.reply_text(
            "❌ Docker CLI not found on this system."
        )

    if not NAVIDROME_MUSIC_DIR.exists() or not NAVIDROME_MUSIC_DIR.is_dir():
        return await update.message.reply_text(
            f"❌ Music directory not found: <code>{html.escape(str(NAVIDROME_MUSIC_DIR))}</code>",
            parse_mode="HTML",
        )

    status_msg = await update.message.reply_text("🎵 Updating playlists...")
    await asyncio.sleep(0.8)

    try:
        playlists_updated, tracks_indexed = _generate_playlists(NAVIDROME_MUSIC_DIR)

        await status_msg.edit_text("🐋 Scanning Navidrome library...")

        scan_output = _scan_navidrome_library(NAVIDROME_APP_DIR)

    except Exception as e:
        log_callback(
            log,
            update.effective_user,
            "navidrome",
            "update_playlist",
            "failed",
            detail=str(e),
        )
        return await status_msg.edit_text(
            f"❌ <b>Update failed</b>\n<code>{html.escape(str(e))}</code>",
            parse_mode="HTML",
        )

    scan_output = tail_log_lines(scan_output, 25)

    text = (
        "✅ <b>Playlist update complete</b>\n\n"
        f"<b>Playlists updated:</b> {playlists_updated}\n"
        f"<b>Tracks indexed:</b> {tracks_indexed:,}\n"
        "\n"
        f"<b>Music dir:</b> <code>{html.escape(str(NAVIDROME_MUSIC_DIR))}</code>\n"
        f"<b>Docker app:</b> <code>{html.escape(NAVIDROME_APP_DIR)}</code>\n\n"
        f"<b>Navidrome scan output</b>\n<pre>{html.escape(scan_output or 'No output.')}</pre>"
    )

    if len(text) > 4000:
        text = (
            "✅ <b>Playlist update complete</b>\n"
            "(Output was too long and has been truncated.)\n\n"
            f"<b>Navidrome scan output</b>\n<pre>{html.escape(tail_log_lines(scan_output, 12))}</pre>"
        )

    log_callback(
        log,
        update.effective_user,
        "navidrome",
        "update_playlist",
        "completed",
        detail=f"playlists={playlists_updated} tracks={tracks_indexed}",
    )
    await status_msg.edit_text(text, parse_mode="HTML")


def get_help_section() -> str | None:
    if not NAVIDROME_PLAYLIST_UPDATE_CMD:
        return None
    return (
        "‣ <code>/update_playlist</code> — Rebuild m3u8 playlists and trigger Navidrome scan"
    )


def get_commands() -> list[tuple[str, str]]:
    if not NAVIDROME_PLAYLIST_UPDATE_CMD:
        return []
    return [
        (
            "update_playlist",
            "Rebuild m3u8 playlists and trigger Navidrome scan",
        )
    ]


def register_handlers(app):
    if not NAVIDROME_PLAYLIST_UPDATE_CMD:
        return

    app.add_handler(CommandHandler("update_playlist", update_playlist))
