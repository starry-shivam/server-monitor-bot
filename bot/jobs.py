import time
import psutil
from pathlib import Path

from bot.config import LOG_CHANNEL_ID

# --- Alert watchdog data ---
last_alert = {"temp": 0.0, "ram": 0.0}


def _get_uptime() -> float:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except Exception:
        return 0.0


async def notify_boot_job(context):
    server_uptime = _get_uptime()
    reason = "server reboot" if server_uptime < 30 else "manual restart"
    await context.bot.send_message(
        chat_id=LOG_CHANNEL_ID,
        text=f"✅ Bot started (reason: {reason})",
    )


async def watchdog_job(context):
    bot = context.bot
    now = time.time()
    temp_c = 0.0

    # Efficiently get first available temperature
    temps = psutil.sensors_temperatures()
    for entries in temps.values():
        for e in entries:
            if e.current:
                temp_c = e.current
                break
        if temp_c:
            break

    mem_pct = psutil.virtual_memory().percent

    # CPU temp alert (65°C) - Cooldown 30 mins
    if temp_c > 65 and (now - last_alert["temp"] > 1800):
        last_alert["temp"] = now
        await bot.send_message(
            chat_id=LOG_CHANNEL_ID,
            text=f"🔥 *High CPU Temp:* `{temp_c:.1f}°C`",
            parse_mode="Markdown",
        )

    # RAM alert (80%) - Cooldown 30 mins
    if mem_pct > 80 and (now - last_alert["ram"] > 1800):
        last_alert["ram"] = now
        await bot.send_message(
            chat_id=LOG_CHANNEL_ID,
            text=f"📈 *High RAM Usage:* `{mem_pct:.1f}%`",
            parse_mode="Markdown",
        )
