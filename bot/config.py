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

import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_IDS = [int(x) for x in os.getenv("OWNER_IDS", "0").split(",") if x.strip()]
LOG_CHANNEL_ID = int(os.getenv("LOG_CHANNEL_ID", "0"))
CALLBACK_TTL = int(os.getenv("CALLBACK_TTL", "300"))  # seconds
CALLBACK_SIG_SECRET = os.getenv("CALLBACK_SIG_SECRET", uuid.uuid4().hex)
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "").strip()
TELEGRAM_API_BASE_URL = os.getenv("TELEGRAM_API_BASE_URL", "").strip().rstrip("/")
POWER_MGMT_AVAILABLE = os.getenv("POWER_MGMT_AVAILABLE", "false").lower() == "true"
NOTIFY_DOCKER_UPDATES = os.getenv("NOTIFY_DOCKER_UPDATES", "false").lower() == "true"
ADDITIONAL_DRIVE_PATHS = [
    x.strip() for x in os.getenv("ADDITIONAL_DRIVE_PATHS", "").split(",") if x.strip()
]

# --- Shell Config ---
SHELL_ALLOWED_COMMANDS = {
    "date",
    "df",
    "ls",
    "cat",
    "fastfetch",
    "free",
    "id",
    "lsblk",
    "ps",
    "pwd",
    "ss",
    "uname",
    "uptime",
    "whoami",
}
SHELL_FORBIDDEN_CHARS = {"&", ";", "|", ">", "<", "$", "`", "\\"}
SHELL_TIMEOUT = int(os.getenv("SHELL_TIMEOUT", "45"))  # seconds
SHELL_MAX_OUTPUT = int(os.getenv("SHELL_MAX_OUTPUT", "3600"))
SHELL_ENABLED = os.getenv("SHELL_ENABLED", "false").lower() == "true"
PYEXEC_ENABLED = os.getenv("PYEXEC_ENABLED", "false").lower() == "true"

# --- Docker action config ---
DC_ALLOWED_ACTIONS = [
    "config",
    "pull",
    "build",
    "up",
    "stop",
    "down",
    "update",
    "prune",
    "logs",
    "restart",
]
DOCKER_APPS_DIR = Path(os.getenv("DOCKER_APPS_DIR", ""))
# Comma-separated list of directory names to ignore
DC_IGNORE_DIRS = [
    x.strip() for x in os.getenv("DC_IGNORE_DIRS", "").split(",") if x.strip()
]
DC_IGNORE_UPDATE_NOTIF_DIRS = [
    x.strip()
    for x in os.getenv("DC_IGNORE_UPDATE_NOTIF_DIRS", "").split(",")
    if x.strip()
]
