import re
import subprocess
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes
from bot.auth import restricted

# --- Pre-compiled Regex ---
RE_PMIC_CURRENT = re.compile(r"(\S+)_A.*?=([\d.]+)A")
RE_PMIC_VOLTAGE = re.compile(r"(\S+)_V.*?=([\d.]+)V")
RE_THROTTLE_HEX = re.compile(r"0x([0-9A-Fa-f]+)")


# ================= Hardware Helpers =================


def parse_pmic():
    """Reads Raspberry Pi 5 PMIC ADC rails using vcgencmd."""
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

    lines = [f"⚡ *Raspberry Pi 5 Power Report*\n"]
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


# ================= Handler =================


@restricted
async def powerc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("📡 Reading PMIC ADC…")
    verbose = bool(context.args and "-v" in context.args)
    try:
        report = format_power_report() if verbose else format_minimal_power_report()
        await msg.edit_text(report, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ Error: `{e}`", parse_mode="Markdown")
