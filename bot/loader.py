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
Dynamic module loader.

Automatically discovers, validates, and loads feature and job modules.

Feature modules must export:
  - register_handlers(app): Registers command/callback handlers with the app
  - get_help_section(): Returns help text (str) or None if disabled

Job modules must export:
  - register_jobs(job_queue): Registers jobs with the job queue
"""

import importlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

FEATURES_DIR = Path(__file__).parent / "features"
JOBS_DIR = Path(__file__).parent / "jobs"
REQUIRED_FEATURE_EXPORTS = {"register_handlers", "get_help_section", "get_commands"}
REQUIRED_JOB_EXPORTS = {"register_jobs"}


def _is_valid_module(module, required_exports) -> bool:
    """Check if module exports required functions."""
    for attr_name in required_exports:
        if not hasattr(module, attr_name):
            return False
        attr = getattr(module, attr_name)
        if not callable(attr):
            return False
    return True


def load_feature_modules() -> list[tuple[str, callable, callable, callable]]:
    """
    Dynamically discover and load all feature modules.

    Returns:
        List of tuples: (module_name, register_handlers_func, get_help_section_func, get_commands_func)
    """
    modules = []

    if not FEATURES_DIR.exists():
        log.error(f"Features directory not found: {FEATURES_DIR}")
        return modules

    # Find all .py files in features directory (exclude __init__.py)
    feature_files = sorted(
        f
        for f in FEATURES_DIR.glob("*.py")
        if f.name != "__init__.py" and not f.name.startswith("_")
    )

    for feature_file in feature_files:
        module_name = feature_file.stem
        full_module_name = f"bot.features.{module_name}"

        try:
            module = importlib.import_module(full_module_name)

            if not _is_valid_module(module, REQUIRED_FEATURE_EXPORTS):
                log.warning(
                    f"Module {module_name} missing required exports (register_handlers, get_help_section, get_commands), skipping"
                )
                continue

            register_func = getattr(module, "register_handlers")
            help_func = getattr(module, "get_help_section")
            commands_func = getattr(module, "get_commands")

            modules.append((module_name, register_func, help_func, commands_func))

        except Exception as e:
            log.error(f"Failed to load feature module {module_name}: {e}")
            continue

    return modules


def load_job_modules() -> list[tuple[str, callable]]:
    """
    Dynamically discover and load all job modules.

    Returns:
        List of tuples: (module_name, register_jobs_func)
    """
    modules = []

    if not JOBS_DIR.exists():
        log.error(f"Jobs directory not found: {JOBS_DIR}")
        return modules

    # Find all .py files in jobs directory (exclude __init__.py)
    job_files = sorted(
        f
        for f in JOBS_DIR.glob("*.py")
        if f.name != "__init__.py" and not f.name.startswith("_")
    )

    for job_file in job_files:
        module_name = job_file.stem
        full_module_name = f"bot.jobs.{module_name}"

        try:
            module = importlib.import_module(full_module_name)

            if not _is_valid_module(module, REQUIRED_JOB_EXPORTS):
                log.warning(
                    f"Module {module_name} missing required export (register_jobs), skipping"
                )
                continue

            register_func = getattr(module, "register_jobs")
            modules.append((module_name, register_func))

        except Exception as e:
            log.error(f"Failed to load job module {module_name}: {e}")
            continue

    return modules


def register_all_handlers(app) -> int:
    """
    Register handlers for all discovered feature modules.

    Args:
        app: Telegram ApplicationBuilder instance

    Returns:
        Number of modules successfully registered
    """
    modules = load_feature_modules()
    count = 0

    for module_name, register_func, _, _ in modules:
        try:
            register_func(app)
            count += 1
        except Exception as e:
            log.error(f"Failed to register handlers for {module_name}: {e}")

    return count


def collect_help_sections() -> list[str]:
    """
    Collect help sections from all active feature modules.

    Returns:
        List of help text strings from modules that are active
    """
    modules = load_feature_modules()
    help_sections = []

    for module_name, _, help_func, _ in modules:
        try:
            help_text = help_func()
            if help_text:
                help_sections.append(help_text)
        except Exception as e:
            log.error(f"Failed to get help from {module_name}: {e}")

    return help_sections


def collect_all_commands() -> list[tuple[str, str]]:
    """
    Collect all bot commands from all active feature modules.

    Returns:
        List of (command, description) tuples
    """
    modules = load_feature_modules()
    commands = []

    for module_name, _, _, commands_func in modules:
        try:
            module_commands = commands_func()
            if module_commands:
                commands.extend(module_commands)
        except Exception as e:
            log.error(f"Failed to get commands from {module_name}: {e}")

    return commands


async def set_bot_commands(app) -> bool:
    """
    Set bot commands for the current Telegram bot.

    Collects all commands from feature modules and registers them with Telegram.

    Args:
        app: Telegram Application instance

    Returns:
        True if successful, False otherwise
    """
    try:
        from telegram import BotCommand

        commands = collect_all_commands()

        # Add core commands (always available)
        commands.extend(
            [
                ("start", "Start the bot"),
                ("help", "Show available commands"),
                ("ping", "Measure Telegram API latency"),
            ]
        )

        # Convert to BotCommand objects
        bot_commands = [BotCommand(cmd, desc) for cmd, desc in commands]

        # Set bot commands via Telegram API
        await app.bot.set_my_commands(bot_commands)
        log.info(f"Registered {len(bot_commands)} bot commands")
        return True

    except Exception as e:
        log.error(f"Failed to set bot commands: {e}")
        return False


def register_all_jobs(job_queue) -> int:
    """
    Register jobs for all discovered job modules.

    Args:
        job_queue: Telegram JobQueue instance

    Returns:
        Number of job modules successfully registered
    """
    modules = load_job_modules()
    count = 0

    for module_name, register_func in modules:
        try:
            register_func(job_queue)
            count += 1
        except Exception as e:
            log.error(f"Failed to register jobs for {module_name}: {e}")

    return count
