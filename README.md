# 🫐 Raspberrypi Server Monitor Bot

Lightweight Telegram bot to monitor and manage a **Raspberry Pi home server** (tested on **Raspberry Pi 5**).
Provides system information, resource charts, Docker container listings (text or image), remote shell/Python execution,
Pi-specific power/thermal telemetry, **server reboot notifications**, **daily health reports**, and **automatic alerts**
for high CPU temperature and RAM usage.

Requires `fastfetch` to be installed for `/info` to work.

## Commands
(All commands except `/start` are owner-restricted. Set `OWNER_IDS` to your numeric Telegram user ID.)

- /start — Welcome message
- /help — Show available commands (owner-only)

#### System info
- /info — System info (Neofetch/fastfetch-style; requires `fastfetch`)
- /info -ip — Include local IP in the info block
- /ip — Show server public IP

#### Monitoring
- /cputemp — CPU temperature (Raspberry Pi only)
- /powerc — Raspberry Pi 5 power usage / fan state / voltage / throttle flags
- /stats — Static CPU/RAM/Disk usage chart
- /stats -live — Live system monitor (10 updates with 1s refresh)
- /ping — Measure Telegram API latency and show uptime

#### Docker
- /dockerps — Show Docker containers in formatted text (if it fits)
- /dockerps -img — Force an image-rendered table of containers

#### Remote execution
- /shell <cmd> — Run shell commands on the server
- /pyexec <python> — Execute a Python snippet remotely

#### Behavior & alerts
- Sends “Bot started (likely server reboot)” on startup
- Daily health report (fastfetch summary + minimal power/temperature line)
- Alert if CPU temp > 65 °C (2h cooldown between alerts)
- Alert if RAM usage > 80% (2h cooldown between alerts)
---

## Prerequisites
- Python 3.9+ (3.10 recommended)
- System packages for building wheels if needed (build-essential, libffi-dev, etc.)
- A Telegram bot token and your Telegram numeric user id

## Install & run (local)
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export BOT_TOKEN="123:ABC..."
export OWNER_ID="123456789"
export OWNER_USERNAME="myuser"
python3 bot.py
```

## Run as a systemd service (recommended)
1. Copy and edit the template file `server-monitor-bot/server-monitor-bot.service`
2. Set `User`,`WorkingDirectory`, `ExecStart`, and environment vars (`BOT_TOKEN`, `OWNER_ID`, `OWNER_USERNAME`).
3. Install and enable:

```bash
sudo cp server-monitor-bot.service /etc/systemd/system/server-monitor-bot.service
sudo systemctl daemon-reload
sudo systemctl enable --now server-monitor-bot.service
sudo systemctl status server-monitor-bot.service
```

To view logs:
```bash
journalctl -u server-monitor-bot.service -f
```

## Security notes
- `/shell` and `/pyexec` allow arbitrary code execution — use only in trusted environments.
- Owner-only decorator enforces `OWNER_IDS`; ensure `OWNER_IDS` is set correctly.
- Review and adapt the systemd unit for additional hardening as needed.

## Example Output

<img width="1549" height="860" alt="{89858651-A38A-41B5-A1F0-4C92FB7376C9}" src="https://github.com/user-attachments/assets/9ac39e78-edb3-4e7d-ba43-ec3173c97ffd" />


## License

```
MIT License

Copyright (c) [2025 - Present] Stɑrry Shivɑm

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
