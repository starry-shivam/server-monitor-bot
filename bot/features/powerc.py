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
# Dragon Q6A Power / Thermal Monitor

import hmac
import logging
import os
import time
from pathlib import Path
from collections import OrderedDict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.auth import restricted, is_authorized_callback_user
from bot.logger import log_callback, log_security_event
from bot.features import cb_sign

log = logging.getLogger(__name__)

# =====================================================
# Refresh Control
# =====================================================

POWERC_REFRESH_COOLDOWN = 5
_POWERC_REFRESH_TS = OrderedDict()
_POWERC_MAX_CACHE_SIZE = 15


# =====================================================
# Hardware Detection
# =====================================================


def is_dragon_q6a():
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00")
        return "Dragon Q6A" in model or "Radxa Dragon Q6A" in model
    except Exception:
        return False


# =====================================================
# Helpers
# =====================================================


def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def read_float(path):
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return None


# =====================================================
# Temperature Sensors
# =====================================================


def thermal_zones():
    zones = []

    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            t = (z / "type").read_text().strip()
            temp = int((z / "temp").read_text().strip()) / 1000.0
            zones.append((t, temp))
        except Exception:
            continue

    return zones


def get_cpu_temp():
    vals = []

    for name, temp in thermal_zones():
        if name.startswith("cpu") or name.startswith("cpuss"):
            vals.append(temp)

    if not vals:
        return 0.0

    return sum(vals) / len(vals)


def get_cpu_hotspot():
    vals = []

    for name, temp in thermal_zones():
        if name.startswith("cpu") or name.startswith("cpuss"):
            vals.append(temp)

    return max(vals) if vals else 0.0


def get_named_temp(prefix):
    vals = [temp for name, temp in thermal_zones() if name.startswith(prefix)]
    return max(vals) if vals else None


# =====================================================
# CPU Frequency
# =====================================================


def cpu_freqs():
    out = []

    for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        cur = read_int(cpu / "cpufreq/scaling_cur_freq")
        mx = read_int(cpu / "cpufreq/cpuinfo_max_freq")

        if cur and mx:
            out.append((cpu.name, cur, mx))

    return out


def group_freqs():
    cpus = cpu_freqs()

    little = []
    mid = []
    prime = []

    for name, cur, mx in cpus:
        idx = int(name.replace("cpu", ""))

        if idx <= 3:
            little.append((cur, mx))
        elif idx <= 6:
            mid.append((cur, mx))
        else:
            prime.append((cur, mx))

    def avg(lst):
        if not lst:
            return (0, 0)
        return (
            int(sum(x[0] for x in lst) / len(lst)),
            int(sum(x[1] for x in lst) / len(lst)),
        )

    return {
        "little": avg(little),
        "mid": avg(mid),
        "prime": avg(prime),
    }


# =====================================================
# Load
# =====================================================


def get_load():
    try:
        return os.getloadavg()
    except Exception:
        return (0.0, 0.0, 0.0)


# =====================================================
# Thermal State
# =====================================================


def get_thermal_state():
    temp = get_cpu_hotspot()

    if temp >= 85:
        return "🔴 HOT (Possible throttling)"
    elif temp >= 75:
        return "🟠 Warm"
    else:
        return "🟢 Normal"


# =====================================================
# Formatting
# =====================================================


def mhz_to_ghz(v):
    return v / 1_000_000


def format_minimal_power_report():
    cpu = get_cpu_temp()
    hotspot = get_cpu_hotspot()
    grp = group_freqs()
    nvme = get_named_temp("nvme")
    state = get_thermal_state()

    prime_cur, prime_max = grp["prime"]

    lines = [
        "⚡ *Dragon Q6A*",
        f"🌡 CPU: `{cpu:.1f}°C` (max `{hotspot:.1f}°C`)",
        f"⚙ Prime: `{mhz_to_ghz(prime_cur):.2f}/{mhz_to_ghz(prime_max):.2f} GHz`",
        f"🔥 Status: {state}",
    ]

    if nvme:
        lines.append(f"💽 NVMe: `{nvme:.1f}°C`")

    return "\n".join(lines)


def format_power_report():
    cpu = get_cpu_temp()
    hotspot = get_cpu_hotspot()
    gpu = get_named_temp("gpu")
    ddr = get_named_temp("ddr")
    skin = get_named_temp("msm-skin")
    nvme = get_named_temp("nvme")

    grp = group_freqs()
    load1, load5, load15 = get_load()
    state = get_thermal_state()

    lcur, lmax = grp["little"]
    mcur, mmax = grp["mid"]
    pcur, pmax = grp["prime"]

    lines = [
        "⚡ *Dragon Q6A Detailed Report*",
        "",
        f"🌡 CPU Avg: `{cpu:.1f}°C`",
        f"🔥 CPU Max: `{hotspot:.1f}°C`",
    ]

    if gpu:
        lines.append(f"🎮 GPU: `{gpu:.1f}°C`")
    if ddr:
        lines.append(f"🧠 DDR: `{ddr:.1f}°C`")
    if skin:
        lines.append(f"📦 Skin: `{skin:.1f}°C`")
    if nvme:
        lines.append(f"💽 NVMe: `{nvme:.1f}°C`")

    lines.extend(
        [
            "",
            "*CPU Frequencies:*",
            f"`Little  {mhz_to_ghz(lcur):.2f}/{mhz_to_ghz(lmax):.2f} GHz`",
            f"`Mid     {mhz_to_ghz(mcur):.2f}/{mhz_to_ghz(mmax):.2f} GHz`",
            f"`Prime   {mhz_to_ghz(pcur):.2f}/{mhz_to_ghz(pmax):.2f} GHz`",
            "",
            f"🚨 Thermal: {state}",
            f"📈 Load: `{load1:.2f} {load5:.2f} {load15:.2f}`",
        ]
    )

    return "\n".join(lines)


# =====================================================
# Callback Data
# =====================================================


def powerc_callback_data(cb_type: str, user_id: int, msg_id: int, verbose: int):
    payload = f"pwc:{cb_type}:{user_id}:{msg_id}:{verbose}"
    return f"{payload}:{cb_sign(payload)}"


def powerc_keyboard(user_id: int, msg_id: int, verbose: bool):
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔁 Refresh",
                    callback_data=powerc_callback_data(
                        "refresh",
                        user_id,
                        msg_id,
                        int(verbose),
                    ),
                )
            ]
        ]
    )


# =====================================================
# Main Command
# =====================================================


@restricted
async def powerc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_dragon_q6a():
        return await update.message.reply_text(
            "❌ This command is only supported on Radxa Dragon Q6A."
        )

    verbose = bool(context.args and "--verbose" in context.args)

    msg = await update.message.reply_text("📡 Reading sensors...")

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


# =====================================================
# Callback Handler
# =====================================================


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
            detail=f"user={getattr(q.from_user, 'id', '?')} expected={uid}",
        )
        return await q.answer("🚫 Unauthorized", show_alert=True)

    payload = ":".join(parts[:-1])

    if not hmac.compare_digest(sig, cb_sign(payload)):
        log_security_event(log, "powerc_callback", "invalid_signature")
        return await q.answer("🚫 Invalid signature", show_alert=True)

    now = int(time.time())
    last = _POWERC_REFRESH_TS.get(msg_id, 0)
    wait = POWERC_REFRESH_COOLDOWN - (now - last)

    if wait > 0:
        return await q.answer(f"⏳ Wait {wait}s")

    _POWERC_REFRESH_TS[msg_id] = now
    _POWERC_REFRESH_TS.move_to_end(msg_id)

    if len(_POWERC_REFRESH_TS) > _POWERC_MAX_CACHE_SIZE:
        _POWERC_REFRESH_TS.popitem(last=False)

    await q.answer("🔄 Refreshing...")
    await q.edit_message_text("📡 Reading sensors...")

    try:
        report = format_power_report() if verbose else format_minimal_power_report()

        log_callback(
            log,
            q.from_user,
            "powerc",
            cb,
            "executed",
            detail=f"verbose={verbose}",
        )

        await q.edit_message_text(
            report,
            parse_mode="Markdown",
            reply_markup=powerc_keyboard(uid, msg_id, verbose),
        )

    except Exception as e:
        log_callback(
            log,
            q.from_user,
            "powerc",
            cb,
            "failed",
            detail=str(e),
        )

        await q.edit_message_text(
            f"❌ Error: `{e}`",
            parse_mode="Markdown",
        )
