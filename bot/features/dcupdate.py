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
import logging
from urllib.parse import urlparse

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, CommandHandler

from bot.auth import restricted
from bot.config import DOCKER_APPS_DIR, DC_IGNORE_DIRS
from bot.features.dcaction import list_docker_dirs, has_compose_file
from bot.logger import log_callback

log = logging.getLogger(__name__)

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


def _get_image_source_url(image_tag: str) -> str | None:
    try:
        proc = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image_tag,
                "--format={{json .Config.Labels}}",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip() not in ("", "null"):
            labels = json.loads(proc.stdout)
            if isinstance(labels, dict):
                src = labels.get("org.opencontainers.image.source")
                if isinstance(src, str) and src.strip():
                    return src.strip()
    except Exception:
        pass

    return None


def _to_github_changelog_url(source_url: str | None) -> str | None:
    if not source_url:
        return None

    cleaned = source_url.strip().rstrip("/")
    if cleaned.startswith("git+"):
        cleaned = cleaned[4:]
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]

    try:
        parsed = urlparse(cleaned)
        if parsed.netloc.lower() != "github.com":
            return None
        path = parsed.path.strip("/")
        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
        return f"https://github.com/{owner}/{repo}/releases"
    except Exception:
        return None


def get_github_changelog_url(image_tag: str) -> str | None:
    source = _get_image_source_url(image_tag)
    return _to_github_changelog_url(source)


def check_dir_updates(dir_name: str, system_arch: str) -> list[dict[str, str | None]]:
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

    updates_found: list[dict[str, str | None]] = []

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
                updates_found.append(
                    {
                        "name": name,
                        "image": image_tag,
                        "github_changelog": get_github_changelog_url(image_tag),
                    }
                )

        except Exception:
            continue

    return updates_found


def has_dir_updates(dir_name: str, system_arch: str | None = None) -> bool:
    """Helper for checking if a specific app directory has updates.

    Pass *system_arch* to reuse a previously computed architecture string
    and avoid redundant ``get_system_arch()`` calls in bulk flows.
    """
    arch = system_arch if system_arch is not None else get_system_arch()
    return bool(check_dir_updates(dir_name, arch))


# ================= Manual Command Handler =================


@restricted
async def dcupdate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    user_msg = update.message
    user = update.effective_user

    if not shutil.which("docker"):
        return await update.message.reply_text(
            "❌ Docker CLI not found on this system."
        )

    log_callback(log, user, "dcupdate", "run", "accepted")

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
        log_callback(log, user, "dcupdate", "run", "failed", detail=str(e))
        await status_msg.edit_text(
            f"❌ <b>Error:</b>\n<code>{str(e)}</code>", parse_mode="HTML"
        )
        return

    if not results:
        final_text = "✅ <b>All containers are up to date.</b>"
        reply_markup = None
    else:
        header = "Container updates available"
        if len(results) == 1:
            header = "Container update available"

        final_text = f"📦 <b>{header}</b>\n\n"
        single_changelog_button: InlineKeyboardButton | None = None
        is_single = len(results) == 1

        for app, updates in results.items():
            image = (updates[0].get("image") if updates else None) or "unknown"
            line = f"‣ <b>{app}</b> (<code>{image}</code>)"

            changelog_url = updates[0].get("github_changelog") if updates else None
            if is_single and changelog_url:
                single_changelog_button = InlineKeyboardButton(
                    "📝 Changelog", url=changelog_url
                )
            elif (not is_single) and changelog_url:
                line += f' - <a href="{changelog_url}">changelog</a>'

            final_text += f"{line}\n"
        final_text += "\n"

        if len(results) == 1:
            app_name = next(iter(results))
            final_text += (
                f"Run <code>/dcaction update {app_name}</code> to update this app."
            )
            if single_changelog_button:
                reply_markup = InlineKeyboardMarkup([[single_changelog_button]])
            else:
                reply_markup = None
        else:
            final_text += "Run <code>/dcaction update &lt;dir&gt;</code> to update the specified app."
            reply_markup = None

    log_callback(
        log,
        user,
        "dcupdate",
        "run",
        "completed",
        detail=f"targets={len(targets)} updates={len(results)}",
    )
    await status_msg.edit_text(
        final_text,
        parse_mode="HTML",
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )


def get_help_section() -> str:
    return "‣ <code>/dcupdate</code> — Check for Docker container updates"


def get_commands() -> list[tuple[str, str]]:
    return [("dcupdate", "Check for Docker container updates")]


def register_handlers(app):
    app.add_handler(CommandHandler("dcupdate", dcupdate))
