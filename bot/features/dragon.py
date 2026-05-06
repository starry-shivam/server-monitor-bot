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
# Radxa Dragon Q6A Hardware Monitor
# /dragon   -> compact
# /dragon --verbose -> full hardware report

import hmac
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from collections import OrderedDict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler

from bot.auth import restricted, is_authorized_callback_user
from bot.logger import log_callback, log_security_event
from bot.features import cb_sign

log = logging.getLogger(__name__)

# =====================================================
# Refresh Control
# =====================================================

DRAGON_REFRESH_COOLDOWN = 5
_DRAGON_REFRESH_TS = OrderedDict()
_DRAGON_MAX_CACHE_SIZE = 15


# =====================================================
# Detection
# =====================================================


def is_dragon_q6a():
    try:
        model = Path("/proc/device-tree/model").read_text(errors="ignore").strip("\x00")
        return "Dragon Q6A" in model or "Radxa Dragon Q6A" in model
    except Exception:
        return False


# =====================================================
# Helpers
# =====================================================


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def read_int(path):
    try:
        return int(read_text(path))
    except Exception:
        return None


def run_cmd(cmd):
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def ghz(khz):
    return khz / 1_000_000 if khz else 0.0


def human_bytes(v):
    units = ["B", "K", "M", "G", "T"]
    n = float(v)

    for u in units:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024

    return f"{n:.1f}P"


# =====================================================
# Uptime / RAM / Governor
# =====================================================


def get_uptime():
    try:
        secs = float(Path("/proc/uptime").read_text().split()[0])

        days = int(secs // 86400)
        hours = int((secs % 86400) // 3600)
        mins = int((secs % 3600) // 60)

        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m"

    except Exception:
        return "Unknown"


def get_governor():
    for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        gov = read_text(cpu / "cpufreq/scaling_governor")
        if gov:
            return gov
    return "Unknown"


def get_ram():
    try:
        info = {}

        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0])

        total = info["MemTotal"] * 1024
        avail = info["MemAvailable"] * 1024
        used = total - avail
        pct = used / total * 100

        return human_bytes(used), human_bytes(total), pct

    except Exception:
        return ("?", "?", 0)


# =====================================================
# Thermal
# =====================================================


def thermal_zones():
    rows = []

    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            name = read_text(z / "type")
            temp = read_int(z / "temp")

            if name and temp is not None:
                rows.append((name, temp / 1000.0))
        except Exception:
            continue

    return rows


def temps_by_prefix(prefixes):
    vals = []

    for name, temp in thermal_zones():
        for p in prefixes:
            if name.startswith(p):
                vals.append(temp)
                break

    return vals


def cpu_temps():
    return temps_by_prefix(["cpu", "cpuss"])


def cpu_avg():
    vals = cpu_temps()
    return sum(vals) / len(vals) if vals else 0.0


def cpu_max():
    vals = cpu_temps()
    return max(vals) if vals else 0.0


def named_temp(prefix):
    vals = temps_by_prefix([prefix])
    return max(vals) if vals else None


def thermal_state():
    t = cpu_max()

    if t >= 85:
        return "🔴 HOT (Possible throttling)"
    elif t >= 75:
        return "🟠 Warm"
    elif t >= 65:
        return "🟡 Elevated"
    return "🟢 Normal"


# =====================================================
# CPU Clocks
# cpu0-3 Efficiency
# cpu4-6 Performance
# cpu7   Prime
# =====================================================


def cpu_freqs():
    rows = []

    for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        try:
            idx = int(cpu.name.replace("cpu", ""))
        except Exception:
            continue

        cur = read_int(cpu / "cpufreq/scaling_cur_freq")
        mx = read_int(cpu / "cpufreq/cpuinfo_max_freq")

        if cur and mx:
            rows.append((idx, cur, mx))

    return sorted(rows)


def avg_pairs(items):
    if not items:
        return (0, 0)

    return (
        int(sum(x[0] for x in items) / len(items)),
        int(sum(x[1] for x in items) / len(items)),
    )


def cluster_freqs():
    eff = []
    perf = []
    prime = []

    for idx, cur, mx in cpu_freqs():
        if idx <= 3:
            eff.append((cur, mx))
        elif idx <= 6:
            perf.append((cur, mx))
        elif idx == 7:
            prime.append((cur, mx))

    return {
        "efficiency": avg_pairs(eff),
        "performance": avg_pairs(perf),
        "prime": avg_pairs(prime),
    }


def cpu_count():
    return len(cpu_freqs()) or (os.cpu_count() or 8)


# =====================================================
# Load
# =====================================================


def load_info():
    try:
        l1, l5, l15 = os.getloadavg()
    except Exception:
        l1, l5, l15 = 0.0, 0.0, 0.0

    cores = cpu_count()
    ratio = l1 / cores if cores else 0

    if ratio < 0.25:
        state = "🟢 Light"
    elif ratio < 0.50:
        state = "🟡 Moderate"
    elif ratio < 0.80:
        state = "🟠 Heavy"
    else:
        state = "🔴 Saturated"

    return state, l1, l5, l15, cores


# =====================================================
# USB / PCIe
# =====================================================


def usb_devices():
    out = run_cmd(["lsusb"])
    lines = []

    for row in out.splitlines():
        if "root hub" in row.lower():
            continue

        if "Terminus" in row:
            lines.append("USB Hub")
        elif "AICSemi" in row:
            lines.append("AICSemi AIC8800D80")
        else:
            lines.append(row.split("ID")[-1].strip())

    return lines[:6]


def pcie_devices():
    out = run_cmd(["lspci", "-nn"])
    lines = []

    for row in out.splitlines():
        low = row.lower()

        if "root complex" in low or "pci bridge" in low:
            continue

        if "realtek" in low:
            lines.append("Realtek Gigabit Ethernet")
        elif "nvme" in low or "non-volatile memory" in low:
            lines.append("KingSpec NVMe SSD")
        else:
            lines.append(row.split(": ", 1)[-1])

    return lines[:6]


# =====================================================
# Storage
# =====================================================


def mount_usage(path):
    try:
        return shutil.disk_usage(path)
    except Exception:
        return None


def storage_lines():
    lines = []

    root = mount_usage("/")
    if root:
        total, used, free = root
        lines.append(f"Root: {human_bytes(free)} free / {human_bytes(total)}")

    ssd_path = "/home/starry/ssd"
    if Path(ssd_path).exists():
        ssd = mount_usage(ssd_path)
        if ssd:
            total, used, free = ssd
            lines.append(f"NVMe: {human_bytes(free)} free / {human_bytes(total)}")

    return lines


# =====================================================
# Formatting
# =====================================================


def format_compact():
    avg = cpu_avg()
    mx = cpu_max()

    freq = cluster_freqs()
    pcur, pmax = freq["prime"]

    load_state_txt, l1, _, _, cores = load_info()

    return "\n".join(
        [
            "🐉 *Dragon Q6A*\n",
            f"🌡 CPU: `{avg:.1f}°C` (max `{mx:.1f}°C`)",
            f"⚙ Prime: `{ghz(pcur):.2f}/{ghz(pmax):.2f} GHz`",
            f"🔥 Thermal: {thermal_state()}",
            f"📈 Load: {load_state_txt} (`{l1:.2f}` / `{cores} cores`)",
            f"⏱ Uptime: `{get_uptime()}`",
        ]
    )


def format_verbose():
    avg = cpu_avg()
    mx = cpu_max()

    gpu = named_temp("gpu")
    ddr = named_temp("ddr")
    skin = named_temp("msm-skin")
    nvme_temp = named_temp("nvme")

    gov = get_governor()
    up = get_uptime()
    ram_used, ram_total, ram_pct = get_ram()

    freq = cluster_freqs()
    ecur, emax = freq["efficiency"]
    pcur, pmax = freq["performance"]
    xcur, xmax = freq["prime"]

    load_state_txt, l1, l5, l15, cores = load_info()

    lines = [
        "🐉 *Dragon Q6A Full Report*",
        "",
        f"🌡 CPU Avg: `{avg:.1f}°C`",
        f"🔥 CPU Max: `{mx:.1f}°C`",
    ]

    if gpu is not None:
        lines.append(f"🎮 GPU: `{gpu:.1f}°C`")
    if ddr is not None:
        lines.append(f"🧠 DDR: `{ddr:.1f}°C`")
    if skin is not None:
        lines.append(f"📦 Skin: `{skin:.1f}°C`")
    if nvme_temp is not None:
        lines.append(f"💽 NVMe Temp: `{nvme_temp:.1f}°C`")

    lines.extend(
        [
            "",
            f"⚙ Governor: `{gov}`",
            f"⏱ Uptime: `{up}`",
            f"🧠 RAM: `{ram_used}/{ram_total}` (`{ram_pct:.0f}%`)",
            "",
            "*CPU Frequencies:*",
            f"`Efficiency   {ghz(ecur):.2f}/{ghz(emax):.2f} GHz`",
            f"`Performance {ghz(pcur):.2f}/{ghz(pmax):.2f} GHz`",
            f"`Prime       {ghz(xcur):.2f}/{ghz(xmax):.2f} GHz`",
            "",
            f"🚨 Thermal: {thermal_state()}",
            f"📈 Load: {load_state_txt}",
            f"`1m {l1:.2f} | 5m {l5:.2f} | 15m {l15:.2f}`",
            f"`{cores} CPU cores detected`",
        ]
    )

    stores = storage_lines()
    if stores:
        lines.append("")
        lines.append("*💾 Storage:*")
        for s in stores:
            lines.append(f"• {s}")

    pcie = pcie_devices()
    if pcie:
        lines.append("")
        lines.append("*🧩 PCIe Devices:*")
        for d in pcie:
            lines.append(f"• {d}")

    usb = usb_devices()
    if usb:
        lines.append("")
        lines.append("*🔌 USB Devices:*")
        for d in usb:
            lines.append(f"• {d}")

    return "\n".join(lines)


# =====================================================
# Callback Helpers
# =====================================================


def dragon_callback_data(cb_type, user_id, msg_id, verbose):
    payload = f"drg:{cb_type}:{user_id}:{msg_id}:{verbose}"
    return f"{payload}:{cb_sign(payload)}"


def dragon_keyboard(user_id, msg_id, verbose):
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔁 Refresh",
                    callback_data=dragon_callback_data(
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
# Command
# =====================================================


@restricted
async def dragon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    verbose = bool(context.args and "--verbose" in context.args)

    msg = await update.message.reply_text("📡 Reading sensors...")

    try:
        report = format_verbose() if verbose else format_compact()

        await msg.edit_text(
            report,
            parse_mode="Markdown",
            reply_markup=dragon_keyboard(
                update.effective_user.id,
                msg.message_id,
                verbose,
            ),
        )

    except Exception as e:
        await msg.edit_text(
            f"❌ Error: `{e}`",
            parse_mode="Markdown",
        )


# =====================================================
# Callback
# =====================================================


async def dragon_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    parts = q.data.split(":")

    if len(parts) != 6 or parts[0] != "drg":
        log_security_event(log, "dragon_callback", "invalid_payload")
        return await q.answer("🚫 Invalid callback", show_alert=True)

    _, cb, uid, msg_id, verbose, sig = parts

    uid = int(uid)
    msg_id = int(msg_id)
    verbose = bool(int(verbose))

    if not is_authorized_callback_user(getattr(q.from_user, "id", None), uid):
        log_security_event(
            log,
            "dragon_callback",
            "blocked",
            detail=f"user={getattr(q.from_user,'id','?')} expected={uid}",
        )
        return await q.answer("🚫 Unauthorized", show_alert=True)

    payload = ":".join(parts[:-1])

    if not hmac.compare_digest(sig, cb_sign(payload)):
        log_security_event(log, "dragon_callback", "invalid_signature")
        return await q.answer("🚫 Invalid signature", show_alert=True)

    now = int(time.time())
    last = _DRAGON_REFRESH_TS.get(msg_id, 0)
    wait = DRAGON_REFRESH_COOLDOWN - (now - last)

    if wait > 0:
        return await q.answer(f"⏳ Wait {wait}s")

    _DRAGON_REFRESH_TS[msg_id] = now
    _DRAGON_REFRESH_TS.move_to_end(msg_id)

    if len(_DRAGON_REFRESH_TS) > _DRAGON_MAX_CACHE_SIZE:
        _DRAGON_REFRESH_TS.popitem(last=False)

    await q.answer("🔄 Refreshing...")
    await q.edit_message_text("📡 Reading sensors...")

    try:
        report = format_verbose() if verbose else format_compact()

        log_callback(
            log,
            q.from_user,
            "dragon",
            cb,
            "executed",
            detail=f"verbose={verbose}",
        )

        await q.edit_message_text(
            report,
            parse_mode="Markdown",
            reply_markup=dragon_keyboard(uid, msg_id, verbose),
        )

    except Exception as e:
        log_callback(
            log,
            q.from_user,
            "dragon",
            cb,
            "failed",
            detail=str(e),
        )

        await q.edit_message_text(
            f"❌ Error: `{e}`",
            parse_mode="Markdown",
        )


def get_help_section() -> str | None:
    if not is_dragon_q6a():
        return None
    return "‣ <code>/dragon</code> — Get Dragon Q6A hardware report"


def get_commands() -> list[tuple[str, str]]:
    if not is_dragon_q6a():
        return []
    return [("dragon", "Get Dragon Q6A hardware report")]


def register_handlers(app):
    if not is_dragon_q6a():
        return
    app.add_handler(CommandHandler("dragon", dragon))
    app.add_handler(CallbackQueryHandler(dragon_callback, pattern=r"^drg:"))
