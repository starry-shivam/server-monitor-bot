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

import re
import subprocess
from pathlib import Path
import time
import hmac
import logging
from collections import OrderedDict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from bot.auth import restricted, is_authorized_callback_user
from bot.logger import log_callback, log_security_event
from bot.features import cb_sign


log = logging.getLogger(__name__)

# --- Pre-compiled Regex ---
RE_PMIC_CURRENT = re.compile(r"(\S+)_A.*?=([\d.]+)A")
RE_PMIC_VOLTAGE = re.compile(r"(\S+)_V.*?=([\d.]+)V")
RE_THROTTLE_HEX = re.compile(r"0x([0-9A-Fa-f]+)")

# ================= Refresh Control =================

POWERC_REFRESH_COOLDOWN = 5  # seconds
_POWERC_REFRESH_TS = OrderedDict()
_POWERC_MAX_CACHE_SIZE = 15


# ================= Hardware Helpers =================


def is_rpi5():
    try:
        model = Path("/proc/device-tree/model").read_text()
        return "Raspberry Pi 5" in model
    except Exception:
        return False


def parse_pmic():
    """
    Reads Raspberry Pi 5 PMIC ADC rails using vcgencmd.
    """
    try:
        out = subprocess.check_output(["vcgencmd", "pmic_read_adc"], text=True)
    except subprocess.CalledProcessError:
        return [], 0.0

    current_map = {}
    voltage_map = {}

    for line in out.splitlines():
        if m := RE_PMIC_CURRENT.search(line):
            current_map[m.group(1)] = float(m.group(2))
        elif m := RE_PMIC_VOLTAGE.search(line):
            voltage_map[m.group(1)] = float(m.group(2))

    results = []
    total = 0.0

    for rail, amps in current_map.items():
        if rail in voltage_map:
            volts = voltage_map[rail]
            watts = amps * volts
            results.append((rail, amps, volts, watts))
            total += watts

    return results, total


def get_temp():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except Exception:
        return 0.0


def get_fan():
    try:
        base = Path("/sys/class/thermal/cooling_device0")
        cur = int((base / "cur_state").read_text())
        mx = int((base / "max_state").read_text())
        return cur, mx
    except Exception:
        return None, None


def get_throttle():
    try:
        return subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
    except Exception:
        return "Unknown"


def decode_throttle(hex_str: str) -> str:
    """
    Decodes the throttling status from the hex string returned by vcgencmd get_throttled.
    """
    m = RE_THROTTLE_HEX.search(hex_str)
    if not m:
        return "Unknown"

    val = int(m.group(1), 16)
    flags = []

    # Mapping of bit to message
    conditions = {
        0: "Under-voltage NOW",
        1: "Frequency capped NOW",
        2: "Currently throttled",
        3: "Soft temperature limit NOW",
    }
    history = {
        16: "Under-voltage occurred",
        17: "Frequency cap occurred",
        18: "Throttle occurred",
        19: "Soft temp limit occurred",
    }

    for bit, msg in conditions.items():
        if val & (1 << bit):
            flags.append(f"🔴 {msg} (bit {bit})")

    for bit, msg in history.items():
        if val & (1 << bit):
            flags.append(f"🟡 {msg} (bit {bit})")

    return "\n".join(flags) if flags else "🟢 All good — no throttling"


# ================= Formatting =================


def format_power_report():
    rails, total = parse_pmic()
    temp = get_temp()
    fan_cur, fan_max = get_fan()
    throttle = get_throttle()
    decoded = decode_throttle(throttle)

    lines = ["⚡ *Raspberry Pi 5 Power Report*\n"]
    lines.append(f"🌡Temperature: `{temp:.1f}°C`")

    if fan_cur is not None and fan_max:
        pct = fan_cur / fan_max * 100
        lines.append(f"🌀 Fan: `{fan_cur}/{fan_max}` (`{pct:.0f}%`)")

    lines.append(f"🚨 Throttle: `{throttle}`")
    lines.append(f"{decoded}\n")

    lines.append("*Rails (A × V = W):*")
    # Sort by Watts descending
    for rail, a, v, w in sorted(rails, key=lambda x: -x[3]):
        lines.append(f"`{rail:<10} {a:>5.3f}A × {v:>5.3f}V = {w:>5.3f}W`")

    lines.append(f"\n🔋 *Total Power*: `{total:.3f} W`")
    return "\n".join(lines)


def format_minimal_power_report():
    _, total = parse_pmic()
    temp = get_temp()
    fan_cur, fan_max = get_fan()
    pct = (fan_cur / fan_max * 100) if fan_max else 0
    return f"Power: `{total:.3f} W` | CPU Temp: `{temp:.1f}°C` | Fan: `{pct:.0f}%`"


# ================= Callback Data =================


def powerc_callback_data(cb_type: str, user_id: int, msg_id: int, verbose: int) -> str:
    payload = f"pwc:{cb_type}:{user_id}:{msg_id}:{verbose}"
    return f"{payload}:{cb_sign(payload)}"


def powerc_keyboard(user_id: int, msg_id: int, verbose: bool) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔁 Refresh",
                    callback_data=powerc_callback_data(
                        "refresh", user_id, msg_id, int(verbose)
                    ),
                )
            ]
        ]
    )


# ================= Handler =================


@restricted
async def powerc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if running on Raspberry Pi 5
    if not is_rpi5():
        return await update.message.reply_text(
            "❌ This command is only supported on Raspberry Pi 5."
        )

    verbose = bool(context.args and "--verbose" in context.args)

    # Send initial message
    msg = await update.message.reply_text("📡 Reading PMIC ADC…")
    try:
        report = format_power_report() if verbose else format_minimal_power_report()
        await msg.edit_text(
            report,
            parse_mode="Markdown",
            reply_markup=powerc_keyboard(
                update.effective_user.id,
                msg.message_id,
                verbose,
            ),
        )
    except Exception as e:
        await msg.edit_text(f"❌ Error: `{e}`", parse_mode="Markdown")


# ================= Callback Handler =================


async def powerc_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    parts = q.data.split(":")

    if len(parts) != 6 or parts[0] != "pwc":
        log_security_event(log, "powerc_callback", "invalid_payload")
        return await q.answer("🚫 Invalid callback", show_alert=True)

    _, cb, uid, msg_id, verbose, sig = parts
    uid = int(uid)
    msg_id = int(msg_id)
    verbose = bool(int(verbose))

    if not is_authorized_callback_user(getattr(q.from_user, "id", None), uid):
        log_security_event(
            log,
            "powerc_callback",
            "blocked",
            detail=f"uid={getattr(q.from_user, 'id', '?')} expected={uid}",
        )
        return await q.answer("🚫 Unauthorized", show_alert=True)

    payload = ":".join(parts[:-1])
    if not hmac.compare_digest(sig, cb_sign(payload)):
        log_security_event(log, "powerc_callback", "invalid_signature")
        return await q.answer("🚫 Invalid signature", show_alert=True)

    if not is_rpi5():
        return await q.answer("❌ Only supported on Raspberry Pi 5.", show_alert=True)

    now = int(time.time())
    last = _POWERC_REFRESH_TS.get(msg_id, 0)
    wait = POWERC_REFRESH_COOLDOWN - (now - last)
    if wait > 0:
        return await q.answer(f"⏳ Wait {wait}s")

    _POWERC_REFRESH_TS[msg_id] = now
    _POWERC_REFRESH_TS.move_to_end(msg_id)
    if len(_POWERC_REFRESH_TS) > _POWERC_MAX_CACHE_SIZE:
        _POWERC_REFRESH_TS.popitem(last=False)

    await q.answer("🔄 Refreshing…")
    await q.edit_message_text("📡 Reading PMIC ADC…")

    try:
        report = format_power_report() if verbose else format_minimal_power_report()
        log_callback(log, q.from_user, "powerc", cb, "executed", detail=f"verbose={verbose}")
        await q.edit_message_text(
            report,
            parse_mode="Markdown",
            reply_markup=powerc_keyboard(uid, msg_id, verbose),
        )
    except Exception as e:
        log_callback(log, q.from_user, "powerc", cb, "failed", detail=str(e))
        await q.edit_message_text(f"❌ Error: `{e}`", parse_mode="Markdown")
