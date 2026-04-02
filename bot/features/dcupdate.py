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
#
# Note: This module depends on another module (dcaction.py) for some shared utilities.

import subprocess
import json
import platform
import shutil

from telegram import Update
from telegram.ext import ContextTypes

from bot.auth import restricted
from bot.config import DOCKER_APPS_DIR, DC_IGNORE_DIRS
from bot.features.dcaction import list_docker_dirs, has_compose_file

# ================= System Helpers =================


def get_system_arch() -> str:
    machine = platform.machine().lower()

    if machine == "x86_64":
        return "amd64"
    elif machine == "aarch64":
        return "arm64"
    elif machine.startswith("arm"):
        # Covers armv7l, etc.
        return "arm"

    return "amd64"  # Default fallback


def get_remote_digest(image_name: str, arch: str) -> str | None:
    try:
        proc = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", image_name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return None

        for line in proc.stdout.splitlines():
            if line.startswith("Digest:"):
                return line.split("Digest:")[1].strip()
        return None
    except Exception:
        return None


def check_dir_updates(dir_name: str, system_arch: str) -> list[str]:
    dir_path = DOCKER_APPS_DIR / dir_name
    if not dir_path.exists() or not has_compose_file(dir_path):
        return []

    try:
        proc = subprocess.run(
            ["docker", "compose", "ps", "-q"],
            cwd=dir_path,
            capture_output=True,
            text=True,
            check=True,
        )
        ids = proc.stdout.strip().splitlines()
    except Exception:
        return []

    if not ids:
        return []

    updates_found = []

    for cid in ids:
        if not cid:
            continue
        try:
            fmt = "{{.Name}}|{{.Config.Image}}"
            info_proc = subprocess.run(
                ["docker", "inspect", f"--format={fmt}", cid],
                capture_output=True,
                text=True,
            )

            if info_proc.returncode != 0:
                continue

            name, image_tag = info_proc.stdout.strip().split("|")
            name = name.lstrip("/")

            if ":" not in image_tag:
                image_tag += ":latest"

            img_proc = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    image_tag,
                    "--format={{json .RepoDigests}}",
                ],
                capture_output=True,
                text=True,
            )

            if img_proc.returncode != 0 or img_proc.stdout.strip() in ("", "null"):
                continue

            local_repo_digests = json.loads(img_proc.stdout)

            if not local_repo_digests:
                continue
            local_digest = local_repo_digests[0].split("@")[-1]
            remote_digest = get_remote_digest(image_tag, system_arch)
            if not remote_digest:
                continue

            if local_digest != remote_digest:
                updates_found.append(f"• <b>{name}</b> ({image_tag})")

        except Exception:
            continue

    return updates_found


def has_dir_updates(dir_name: str) -> bool:
    """Helper for checking if a specific app directory has updates."""
    return bool(check_dir_updates(dir_name, get_system_arch()))


# ================= Manual Command Handler =================


@restricted
async def dcupdate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    user_msg = update.message

    if not shutil.which("docker"):
        return await update.message.reply_text(
            "❌ Docker CLI not found on this system."
        )

    status_msg = await user_msg.reply_text(
        "🔎 <b>Checking registries for updates...</b>", parse_mode="HTML"
    )

    targets = []
    if not args or args[0] == "--all":
        targets = list_docker_dirs()
    else:
        target_dir = args[0]
        if target_dir in DC_IGNORE_DIRS:
            await status_msg.edit_text(
                f"🚫 Directory <code>{target_dir}</code> is ignored.", parse_mode="HTML"
            )
            return

        if (DOCKER_APPS_DIR / target_dir).exists():
            targets = [target_dir]
        else:
            await status_msg.edit_text(
                f"❌ Directory <code>{target_dir}</code> not found.", parse_mode="HTML"
            )
            return

    system_arch = get_system_arch()
    results = {}

    try:
        for app_dir in targets:
            updates = check_dir_updates(app_dir, system_arch)
            if updates:
                results[app_dir] = updates

    except Exception as e:
        await status_msg.edit_text(
            f"❌ <b>Error:</b>\n<code>{str(e)}</code>", parse_mode="HTML"
        )
        return

    if not results:
        final_text = "✅ <b>All containers are up to date.</b>"
    else:
        header = "Container updates available"
        if len(results) == 1:
            header = "Container update available"

        final_text = f"📦 <b>{header}</b>\n\n"

        for app, updates in results.items():
            image = updates[0].split("(")[-1].rstrip(")")
            final_text += f"‣ <b>{app}</b> (<code>{image}</code>)\n"
        final_text += "\n"

        if len(results) == 1:
            app_name = next(iter(results))
            final_text += (
                f"Run <code>/dcaction update {app_name}</code> to update this app."
            )
        else:
            final_text += "Run <code>/dcaction update &lt;dir&gt;</code> to update the specified app."

    await status_msg.edit_text(final_text, parse_mode="HTML")
