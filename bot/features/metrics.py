from io import BytesIO

import psutil
import matplotlib

# Set backend to Agg for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from bot.auth import restricted
from bot.config import LIVE_METRICS_URL


async def _metrics_render_chart(
    cpu_pct: float, mem_pct: float, disk_pct: float
) -> bytes:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=(7.5, 3.5), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    fig.suptitle("System Resource Usage", fontsize=12)

    # Bar chart (CPU + Disk)
    labels = ["CPU", "Disk"]
    values = [cpu_pct, disk_pct]
    ax_bar.bar(labels, values, color=["#4CAF50", "#FFC107"])
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("%")

    for i, v in enumerate(values):
        ax_bar.text(
            i,
            min(100, v + 2),
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Pie chart (Memory)
    used = max(0.0, min(100.0, mem_pct))
    ax_pie.pie(
        [used, max(0.0, 100.0 - used)],
        labels=["Used", "Free"],
        colors=["#2196F3", "#B0BEC5"],
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        pctdistance=0.75,
    )
    ax_pie.set_title("Memory")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


@restricted
async def metrics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    img_bytes = await _metrics_render_chart(cpu, mem, disk)
    keyboard = None

    if LIVE_METRICS_URL:
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="📊 Live metrics",
                        url=LIVE_METRICS_URL,
                    )
                ]
            ]
        )

    await update.message.reply_photo(
        photo=img_bytes,
        caption=f"CPU: {cpu:.1f}% | RAM: {mem:.1f}% | Disk: {disk:.1f}%",
        reply_markup=keyboard,
    )
